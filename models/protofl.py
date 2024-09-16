import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
from utils.finch import FINCH
import numpy as np
import pdb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedHierarchy.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def agg_func(protos):
    """
    Returns the svd of the features.
    """

    for [label, proto_list] in protos.items():
        # if len(proto_list) > 1:
        #     # proto = 0 * proto_list[0].data
        #     # for i in proto_list:
        #     #     proto += i.data
        #     proto, _, _ = np.linalg.svd(proto_list, full_matrices=False)
        #     proto = proto/np.linalg.norm(proto, ord=2, axis=0)
        #     protos[label] = proto / len(proto_list)
        # else:
        #     protos[label] = proto_list[0]
        proto_list_np = [tensor.detach().cpu().numpy() for tensor in proto_list]
        proto_list_np = np.array(proto_list_np).T.astype(float)
        proto, _, _ = np.linalg.svd(proto_list_np, full_matrices=False)
        proto = proto/np.linalg.norm(proto, ord=2, axis=0)
        protos[label] = proto[:, 0:10]   # 取前3个左奇异向量
        # pdb.set_trace()

    return protos

def cal_proto(features_map):
    """
    Returns the svd of the features.
    """
    protos = {}
    for [label, features] in features_map.items():
        proto_list_np = [tensor.detach().cpu().numpy() for tensor in features]
        proto_list_np = np.array(proto_list_np).T.astype(float)
        proto, _, _ = np.linalg.svd(proto_list_np, full_matrices=False)
        # pdb.set_trace()
        proto = proto/np.linalg.norm(proto, ord=2, axis=0)
        if proto.shape[1] < 5:
            protos[label] = None
        else:
            protos[label] = proto[:, 0:5].T   # 取前3个左奇异向量
        # 有部分情况下svd之后只得到2个奇异向量，所以取前3个会导致原型维度不一致，这个问题待研究
        # pdb.set_trace()
    return protos


# A, B为输入的数据矩阵，这个函数的作用是每次找到两个矩阵中最相似的两个向量，并把他们放到对应的位置中
# 最终返回的矩阵A_E, B_E内容上没有变化，但调整了列向量的位置，A_E的第i列和B_E的第i列是对应的关系
# 第0列的向量是两者最相似的向量，而第1列则是剩余向量中最相似的，以此类推
def Eq_Basis(A,B):
    AB=np.arccos(A.T@B)
    A_E=np.zeros((A.shape[0],A.shape[1]))
    B_E=np.zeros((B.shape[0],B.shape[1]))
    for i in range(AB.shape[0]):
        ind = np.unravel_index(np.argmin(AB, axis=None), AB.shape)
        AB[ind[0],:]=AB[:,ind[1]]=0
        A_E[:,i]=A[:,ind[0]]
        B_E[:,i]=B[:,ind[1]]
    return  A_E,B_E


class ProtoFL(FederatedModel):
    NAME = 'protofl'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(ProtoFL, self).__init__(nets_list, args, transform)
        self.global_protos = {}
        self.local_protos = {}
        self.infoNCET = args.infoNCET

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        # 进行原型的全局聚合
        for label in local_protos_list.keys():
            global_proto = None
            n_clients = 0
            for client, proto in local_protos_list[label].items():
                if proto is None:
                    continue
                n_clients += 1
                if global_proto is None:
                    global_proto = copy.deepcopy(proto)
                else:
                    global_proto += proto
            agg_protos_label[label] = global_proto / n_clients
        return agg_protos_label

    def hierarchical_info_loss(self, proto_now, client):
        # f_pos和f_neg分别是正类原型和负类原型，一个类别有多个原型
        # f_pos -> torch.Size([3, 512])

        # 不进行全局聚合的原型
        # loss = {}
        # for (label, proto) in proto_now.items():
        #     pdb.set_trace()
        #     # self.global_protos -> {label: {client: proto}}
        #     f_pos = np.array([value for label_, client_data in self.global_protos.items() if label_ == label 
        #                       for client_, value in client_data.items() if client_ != client])
        #     f_neg = list(np.array([value for label_, client_data in self.global_protos.items() if label_ != label 
        #                       for client_, value in client_data.items() if client_ != client]))
        #     f_neg = np.array(np.stack([torch.from_numpy(arr) for arr in f_neg]))
        #     # pdb.set_trace()
        #     loss[label] = self.calculate_infonce(proto, f_pos, f_neg)

        # return loss

        # 进行全局聚合的原型
        loss = {}
        for (label, proto) in proto_now.items():
            # self.global_protos -> {label: proto}}
            if proto is None:
                loss[label] = 0.0
            else:
                f_pos = np.array([value for label_, value in self.global_protos.items() if label_ == label][0])
                f_neg = list(np.array([value for label_, value in self.global_protos.items() if label_ != label]))
                f_neg = np.array(np.stack([arr for arr in f_neg]))
                loss[label] = self.calculate_infonce(proto, f_pos, f_neg)
        return loss

    def calculate_infonce(self, proto, f_pos, f_neg):
        # 不进行全局聚合的原型
        # pos_l = 0.0
        # neg_l = 0.0
        # for i in range(f_pos.shape[0]):
        #     F, G = Eq_Basis(proto.T, f_pos[i].T)
        #     F_in_G = np.clip(F.T@G, a_min = -1, a_max = +1)
        #     Angle = np.arccos(np.abs(F_in_G))
        #     # sim_angle_min[i,j] =  (180/np.pi)*np.min(Angle) 
        #     pos_l += (180/np.pi)*np.trace(Angle)
        #     # pos_l += (180/np.pi)*np.min(Angle)
        # pos_l /= f_pos.shape[0]
        # # pdb.set_trace()
        # for i in range(f_neg.shape[0]):
        #     F, G = Eq_Basis(proto.T, f_neg[i].T)
        #     F_in_G = np.clip(F.T@G, a_min = -1, a_max = +1)
        #     Angle = np.arccos(np.abs(F_in_G))
        #     # sim_angle_min[i,j] =  (180/np.pi)*np.min(Angle) 
        #     # neg_l += (180/np.pi)*np.min(Angle)
        #     neg_l += (180/np.pi)*np.trace(Angle)
        # neg_l /= f_neg.shape[0]
        # # 正类损失 / (正类损失 + 负类损失)
        # # infonce_loss = -torch.log(pos_l / sum_exp_l)
        # # infonce_loss = torch.log(torch.tensor(pos_l / (pos_l + neg_l)))
        # infonce_loss = pos_l / neg_l
        # # pdb.set_trace()
        # return infonce_loss

        # 进行全局聚合的原型
        pos_l = 0.0
        neg_l = 0.0
        F, G = Eq_Basis(proto.T, f_pos.T)
        F_in_G = np.clip(F.T@G, a_min = -1, a_max = +1)
        Angle = np.arccos(np.abs(F_in_G))
        # sim_angle_min[i,j] =  (180/np.pi)*np.min(Angle) 
        pos_l += (180/np.pi)*np.trace(Angle)
        # pos_l += (180/np.pi)*np.min(Angle)

        for i in range(f_neg.shape[0]):
            F, G = Eq_Basis(proto.T, f_neg[i].T)
            F_in_G = np.clip(F.T@G, a_min = -1, a_max = +1)
            Angle = np.arccos(np.abs(F_in_G))
            # sim_angle_min[i,j] =  (180/np.pi)*np.min(Angle) 
            # neg_l += (180/np.pi)*np.min(Angle)
            neg_l += (180/np.pi)*np.trace(Angle)
        neg_l /= f_neg.shape[0]
        # 正类损失 / (正类损失 + 负类损失)
        # infonce_loss = -torch.log(pos_l / sum_exp_l)
        # infonce_loss = torch.log(torch.tensor(pos_l / (pos_l + neg_l)))
        infonce_loss = pos_l / neg_l    # 损失只计算正向原型的作用
        return infonce_loss

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.global_protos = self.proto_aggregation(self.local_protos)
        # self.global_protos = copy.deepcopy(self.local_protos)
        # pdb.set_trace()
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))  # [0-9]代表按类别区分的全局原型
            # pdb.set_trace()
            # all_f = []
            # mean_f = []
            # for protos_key in all_global_protos_keys:
            #     temp_f = self.global_protos[protos_key]
            #     # pdb.set_trace()
            #     temp_f = torch.cat(temp_f, dim=0).to(self.device)
            #     all_f.append(temp_f.cpu())                      # 不求平均值，把同个类别的多个原型都保存到all_f中
            #     mean_f.append(torch.mean(temp_f, dim=0).cpu())  # 求多个原型的平均值
            # all_f = [item.detach() for item in all_f]           # 从计算图分离，不再进行梯度计算
            # mean_f = [item.detach() for item in mean_f]

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            feature_map = {}
            # 整个数据集进行前向运算得到整个数据集的特征向量
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # pdb.set_trace()
                f = net.features(images)

                for i in range(len(labels)):
                    if labels[i].item() in feature_map:
                        feature_map[labels[i].item()].append(f[i, :])
                    else:
                        feature_map[labels[i].item()] = [f[i, :]]
            
            #TODO 如果global_proto是空的，那么先不更新，第一轮先用来计算global_proto
            # 看看feature_map为什么与数据集大小不一样大
            # pdb.set_trace()
            
            proto_now = cal_proto(feature_map)

            if iter == self.local_epoch - 1:
                #TODO 将本地主向量更新到global_proto中
                for label in proto_now.keys():
                    if label not in self.local_protos:
                        self.local_protos[label] = {}
                    self.local_protos[label][index] = proto_now[label]
                    
            if len(self.global_protos.keys()) == 0:
                loss_proto = 0.0
            else:
                loss_proto = self.hierarchical_info_loss(proto_now, index)
            
            # debug
            # if len(self.global_protos.keys()) == 0:
            #     self.global_protos = {}
            #     for i in range(10):
            #         self.global_protos[i] = torch.randn(3, 512)
            # loss_proto = self.hierarchical_info_loss(proto_now, index)
        
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)

                outputs = net.classifier(f)

                lossCE = criterion(outputs, labels)

                # if len(self.global_protos) == 0:
                #     loss_InfoNCE = 0 * lossCE
                # else:
                #     i = 0
                #     loss_InfoNCE = None

                #     for label in labels:
                #         if label.item() in self.global_protos.keys():
                #             f_now = f[i].unsqueeze(0)
                #             # 计算原型损失
                #             loss_instance = self.hierarchical_info_loss(f_now, label, index, all_global_protos_keys)

                #             if loss_InfoNCE is None:
                #                 loss_InfoNCE = loss_instance
                #             else:
                #                 loss_InfoNCE += loss_instance
                #         i += 1
                #     loss_InfoNCE = loss_InfoNCE / i
                # loss_InfoNCE = loss_InfoNCE
                if len(self.global_protos.keys()) == 0:
                    loss_proto_label = 0.0
                else:
                    loss_proto_label = torch.tensor([loss_proto[label.item()] if label.item() in loss_proto else 0.0 for label in labels], requires_grad=True).mean().to(self.device)
                    # if loss_proto_label == 0.0:
                    #     pdb.set_trace()
                # print('loss_proto_label: ', loss_proto_label)
                loss = lossCE + loss_proto_label
                loss.backward()
                iterator.desc = "Local Pariticipant %d CE = %0.3f,InfoNCE = %0.3f" % (index, lossCE, loss_proto_label)
                optimizer.step()

                # if iter == self.local_epoch - 1:
                #     #TODO 将本地主向量更新到global_proto中
                #     for i in range(len(labels)):
                #         if labels[i].item() in agg_protos_label:
                #             agg_protos_label[labels[i].item()].append(f[i, :])
                #         else:
                #             agg_protos_label[labels[i].item()] = [f[i, :]]

        # 平均本地数据中同一类别的特征，得到局部原型
        # svd_protos = agg_func(agg_protos_label)
        # self.local_protos[index] = svd_protos
