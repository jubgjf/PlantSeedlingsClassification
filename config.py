from models import VGG
from models import ResNet
from models import SENet
import torch


class config(object):
    def __init__(self):
        # 数据的路径
        self.data_path = ''
        self.train_path = self.data_path + ''
        self.dev_path = self.data_path + ''
        self.test_path = self.data_path + ''
        self.data_augmentation = True   # argparse  True or False  default True 必须

        # 训练设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # argparse  cpu or cuda  default cuda 必须
        self.model = None             # argparse  VGG or ResNEt or SENet  default SENet 必须
        self.optimizer = None         # argparse  sgd or adam             default adam 必须
        self.learning_rate = 1e-5     # argparse  default 1e-5   非必须
        self.batch_size = 128         # argparse  default 128 非必须
        self.maxiter_without_improvement = 1000  # 若1000轮没有优化则退出

        # 训练相关路径
        self.save_path = './trains'
        self.model_path = self.save_path + '/models/'
        self.log_dir = self.save_path + '/logs/'





