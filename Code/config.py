from models import VGG
from models import ResNet
from models import SENet
import torch


class config(object):
    def __init__(self, args):
        # 数据的路径
        self.data_path = '../Dataset/'
        self.train_path = self.data_path + 'train/'
        self.dev_path = self.data_path + ''
        self.test_path = self.data_path + 'test/'
        self.data_augmentation = args['aug']

        # 训练设置
        self.device = torch.device('cuda' if torch.cuda.is_available() and args['dev'] == 'cuda' else 'cpu')
        self.model = args['model']
        self.optimizer = args['opt']
        self.learning_rate = args['lr']
        self.batch_size = args['bs']
        self.maxiter_without_improvement = args['mwi']

        # 训练相关路径
        self.save_path = '../Trains'
        self.model_path = self.save_path + '/models/'
        self.log_dir = self.save_path + '/logs/'
