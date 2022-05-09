import argparse
import torch


class Parser:
    """命令行参数解析器"""

    def __init__(self):
        # 接收到的命令行参数，初始值是默认值
        self.args = {
            'aug': True,
            'dev': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model': 'SENet',
            'opt': 'Adam',
            'lr': 1e-5,
            'bs': 128,
            'mwi': 1000
        }

        self.parse()

    def parse(self):
        """解析命令行参数，将参数合法化，并存放到 self.args 中"""

        p = argparse.ArgumentParser(description='基于 PyTorch 实现 VGG/ResNet/SENet 等结构')

        p.add_argument('--aug', type=str, default='True', help='data augmentation (default: True)')
        p.add_argument('--dev', type=str, default='cuda', help='cuda or cpu (default: cuda)')
        p.add_argument('--model', type=str, default='SENet', help='VGG/ResNet/SENet (default: SENet)')
        p.add_argument('--opt', type=str, default='Adam', help='SGD/Adam (default: Adam)')
        p.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
        p.add_argument('--bs', type=int, default=128, help='batch size (default: 128)')
        p.add_argument('--mwi', type=int, default=1000, help='max iter without improvement (default: 1000)')

        args = p.parse_args()

        self.legalise(args)

    def legalise(self, args):
        """命令行参数合法化，对不合法的值使用默认值进行覆盖"""

        if args.aug in ('True', 'False'):
            self.args['aug'] = bool(args.aug)
        else:
            print(f'cannot recognize aug = `{args.aug}`, using default `{str(self.args["aug"])}`')

        if args.dev == 'cuda':
            self.args['dev'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif args.dev == 'cpu':
            self.args['dev'] = torch.device(args.dev)
        else:
            print(f'cannot recognize dev = `{args.dev}`, using default `{str(self.args["dev"])}`')

        if args.model in ('SENet', 'VGG', 'ResNet'):
            self.args['model'] = args.model
        else:
            print(f'cannot recognize model = `{args.model}`, using default `{str(self.args["model"])}`')

        if args.opt in ('Adam', 'SGD'):
            self.args['opt'] = args.opt
        else:
            print(f'cannot recognize opt = `{args.opt}`, using default `{str(self.args["opt"])}`')

        if args.lr > 0:
            self.args['lr'] = args.lr
        else:
            print(f'cannot use lr = `{args.lr}`, using default `{str(self.args["lr"])}`')

        if args.bs > 0:
            self.args['bs'] = args.bs
        else:
            print(f'cannot use bs = `{args.bs}`, using default `{str(self.args["bs"])}`')

        if args.mwi > 0:
            self.args['mwi'] = args.mwi
        else:
            print(f'cannot use mwi = `{args.mwi}`, using default `{str(self.args["mwi"])}`')
