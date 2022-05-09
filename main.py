import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='基于PyTorch实现VGG/ResNet/SENet等结构')

    parser.add_argument('--device', type=str, default='cuda', help='use cuda or cpu (default: cuda)')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs (default: 100)')
    parser.add_argument('--model', type=str, default='VGG', help='select model between VGG/ResNet/SENet (default: VGG)')

    args = parser.parse_args()

    if args.device not in ('cuda', 'cpu'):
        print(f"cannot use `{args.device}` as device, try to use cuda")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"device = {device}")

    if args.epochs <= 0:
        print(f"cannot use `{args.epochs}` as epochs, try to use 100")
        epochs = 100
    else:
        epochs = args.epochs
    print(f"epochs = {epochs}")

    if args.model not in ('VGG', 'ResNet', 'SENet'):
        print(f"cannot use `{args.model}` as model, try to use VGG")
        model = 'VGG'
    else:
        model = args.model
    print(f"model  = {model}")
