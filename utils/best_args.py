best_args = {
    'fl_digits': {

        'fedavg': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },
        'fedprox': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu': 0.01,
        },

        'moon': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'temperature': 0.5,
                'mu':5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'Note': '+ MSE'
        },

        'protofl': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },

        'fedsem': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },

        'fedproto': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedga': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedsr': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        }
    },
    'fl_officecaltech': {

        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'fedprox': {
            'local_lr': 0.01,
            'mu': 0.01,
            'local_batch_size': 64,
        },
        'moon': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'temperature': 0.5,
            'mu': 5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'Note': '+ MSE'
        },

        'protofl': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },

        'fedsem': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },

        'fedproto': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedga': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedsr': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        }
    },
    'fl_pacs': {

        'fedavg': {
                'local_lr': 0.003,
                'local_batch_size': 128,
        },
        'fedprox': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu': 0.01,
        },

        'moon': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'temperature': 0.5,
                'mu':5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'Note': '+ MSE'
        },

        'protofl': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },

        'fedsem': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },

        'fedproto': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedga': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedsr': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedfix': {
            'local_lr': 0.003,
            'local_batch_size': 256,
        }
    },
    'fl_officehome': {

        'fedavg': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },
        'fedprox': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu': 0.01,
        },

        'moon': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'temperature': 0.5,
                'mu':5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'Note': '+ MSE'
        },

        'protofl': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },

        'fedsem': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },

        'fedproto': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedga': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },

        'fedsr': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        }
    }
}
