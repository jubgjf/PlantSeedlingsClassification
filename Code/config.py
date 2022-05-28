import torch


class config(object):
    def __init__(self, args):
        # 数据的路径
        self.data_path = '../Dataset/'
        self.train_path = self.data_path + 'train/'
        self.dev_path = self.data_path + 'train/'
        self.test_path = self.data_path + 'test/'
        self.data_augmentation = args['aug']

        # 训练设置
        self.train_dev_frac = 9  # train 和 dev 的比例
        self.device = torch.device('cuda' if torch.cuda.is_available() and args['dev'] == 'cuda' else 'cpu')
        self.model = args['model']
        self.mode = args['mode']
        self.optimizer = args['opt']
        self.learning_rate = args['lr']
        self.batch_size = args['bs']
        self.epoch = args['epoch']
        self.maxiter_without_improvement = 1000
        self.class_num = 12

        # 训练相关路径
        self.save_path = '../Trains'

        # 模型保存路径
        self.model_saved = self.save_path + '/models/' + self.model + '.pkl'
        self.log_dir = self.save_path + '/logs/'

        # 输出路径
        self.output_path = '../out/' + self.model + '_submission.csv'