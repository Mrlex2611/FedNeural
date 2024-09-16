import pdb
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
from backbone.mi_net import get_mi_model, mi_estimate
from backbone.autoencoder import mycnn
from datasets.pacs import FedLeaPACS
from datasets.officecaltech import FedLeaOfficeCaltech
from datasets.officehome import FedLeaOfficeHome
from datasets.digits import FedLeaDigits
from collections import defaultdict
import torch
import numpy as np

class FedProto(FederatedModel):
    NAME = 'fedproto'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedProto, self).__init__(nets_list,args,transform)
        self.global_proto = {}
        self.dataset = FedLeaOfficeHome

    def ini(self):
        # self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net = mycnn(self.dataset.N_CLASS)
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            # pdb.set_trace()
            self._train_net(i,self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)

        return  None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.7, weight_decay=1e-5)
        # optimizer = optim.Adam(net.parameters(), lr=self.local_lr, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_mse1 = nn.MSELoss()
        criterion_cls.to(self.device)
        criterion_mse1.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = net(images)
                loss_cls = criterion_cls(outputs, labels)

                protos = net.semantic_feature(images)
                # prototype loss
                if len(self.global_proto.keys()) == 0:
                    loss_proto = 0.0
                else:
                    # 根据label生成一个形状和z1相同且位置对应的global_proto，z1中每一项对应一个相应类别的原型
                    # proto_tensor = torch.stack([self.global_proto[label.item()].to(self.device) for label in labels])
                    # # pdb.set_trace()
                    # loss_proto = criterion_mse2(z1, proto_tensor)

                    loss_proto = 0.0
                    for i, label in enumerate(labels):
                        label = label.item()  # 从 tensor 转为 int
                        # prototype = torch.tensor(np.random.randn(512), dtype=torch.float32).to(self.device)
                        prototype = self.global_proto[label].detach().to(self.device)  # 确保原型也在正确的设备上
                        # 计算当前样本特征和对应原型之间的 L2 损失
                        loss_proto = loss_proto + criterion_mse1(protos[i], prototype)

                    # 计算批次的平均损失
                    loss_proto /= len(labels)

                loss = loss_cls + loss_proto

                # if loss > 100:
                #     pdb.set_trace()

                # torch.autograd.set_detect_anomaly(True)
            
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f, loss_cls = %0.3f, loss_proto = %0.3f" \
                                    % (index, loss, loss_cls, loss_proto)
                optimizer.step()

                # if loss > 100:
                #     pdb.set_trace()

                # grads = {}

                # print('='*50 + '输出梯度' + '='*50)
                # for name, param in net.named_parameters():
                #     if param.requires_grad:
                #         grads[name] = param.grad
                # # 输出梯度
                # print(grads)
            
            # scheduler.step()
        
        # calculate prototype
        net.eval()
        # 存储每个类别的表征和计数器
        class_representations = defaultdict(lambda: {'sum': 0, 'count': 0})
        for data, labels in train_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            representations = net.semantic_feature(data)

            # 累加每个类别的表征和计数
            for label, representation in zip(labels, representations):
                if label.item() not in class_representations:
                    class_representations[label.item()]['sum'] = representation
                    class_representations[label.item()]['count'] = 1
                else:
                    class_representations[label.item()]['sum'] = class_representations[label.item()]['sum'].clone() + representation
                    class_representations[label.item()]['count'] = class_representations[label.item()]['count'] + 1

        # 计算每个类别的平均表征作为原型
        net.proto = {label: info['sum'] / info['count'] for label, info in class_representations.items()}

    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all
        else:
        # if freq == None:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]
        
        # 可能只有部分客户端有某一个类的原型，因此要按照比例放大对应的加权权重
        adjust = {}
        for c in range(self.dataset.N_CLASS):
            c_len = 0
            for client in online_clients:
                net = nets_list[client]
                if c in net.proto.keys():
                    c_len += 1
            adjust[c] = 1.0 * c_len / self.len(online_clients)

        first = True
        updated_proto = {}
        for index,net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            if first:
                first = False
                for key in net_para:
                    if key in global_w.keys():
                        global_w[key] = net_para[key].clone() * freq[index]
            else:
                for key in net_para:
                    if key in global_w.keys():
                        global_w[key] = global_w[key] + net_para[key].clone() * freq[index]
            
            # update prototype 
            for key in net.proto:
                if key in updated_proto.keys():
                    updated_proto[key] = updated_proto[key] + net.proto[key].clone() * freq[index] / adjust[key]
                else:
                    updated_proto[key] = net.proto[key].clone() * freq[index] / adjust[key]

        global_net.load_state_dict(global_w)

        for _, net in enumerate(nets_list):
            net_para = net.state_dict()
            net_para.update({k: v for k, v in global_w.items() if k in net_para})
            net.load_state_dict(net_para)
        self.global_proto = updated_proto