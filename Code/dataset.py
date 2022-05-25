import os
import torch.utils.data.dataset as dataset
from PIL import Image
from torchvision.transforms import transforms


def getDataset(config):
    """
    获取 train + dev + test 的 dataset

    Args:
        config: 模型配置

    Returns:
        一个 tuple，内部的元素是 (trainDataset, devDataset, testDataset)

    """

    trainDataset = myDataset(buildData(config, "train"))
    devDataset = myDataset(buildData(config, "dev"))
    testDataset = myDataset(buildData(config, "test"))

    return trainDataset, devDataset, testDataset


class buildData():
    def __init__(self, config, choice='train'):
        """
        读入数据，并数据增强，生成一个数据例

        Args:
            config: 模型配置
            choice: 数据集类型，只能是 "train", "dev", "test" 之一
        """

        # 双向索引 label
        label_int2str = {}  # { 0: "Black-grass", 1: "Charlock", ...}
        label_str2int = {}  # { "Black-grass": 0, "Charlock": 1, ...}

        # TODO choice=='dev' or 'test'
        path = config.train_path
        path_label = []
        for label_int, label_dir in enumerate(os.listdir(path)):
            label_int2str[label_int] = label_dir
            label_str2int[label_dir] = label_int
            image_names = os.listdir(path + label_dir)
            for image_name in image_names:
                path_label.append((path + label_dir + "/" + image_name, label_int))

        # 根据 config 进行数据增强
        aug = self.dataAugment(config)

        # buildData.Data 为 List, List 中每一项为二元组 (tensor, label)
        self.Data = [(aug(Image.open(data[0]).convert('RGB')), data[1]) for data in path_label]
        self.length = len(self.Data)

    def dataAugment(self, config):
        """
        进行数据增强

        Args:
            config: 模型配置

        Returns:
            返回 transforms.Compose
        """

        trans = []
        if config.data_augmentation == 'None':
            pass
        else:
            if 'rot' in config.data_augmentation:
                # TODO 旋转会导致图片出现大量的 0，不知道为什么，需要修一下
                trans.append(transforms.RandomRotation(degrees=(0, 180)))  # 随机旋转
            if 'flp' in config.data_augmentation:
                trans.append(transforms.RandomHorizontalFlip())  # 随机水平翻转
                trans.append(transforms.RandomVerticalFlip())  # 随机垂直翻转
            if 'gsc' in config.data_augmentation:
                trans.append(transforms.RandomGrayscale())  # 随机转灰度图
            if 'pst' in config.data_augmentation:
                trans.append(transforms.RandomPosterize(bits=2))  # 随机分色
            if 'slt' in config.data_augmentation:
                trans.append(transforms.RandomSolarize(threshold=192.0))  # 随机曝光
        trans.append(transforms.Resize(100))  # 缩放到 100x100
        trans.append(transforms.ToTensor())

        return transforms.Compose(trans)


class myDataset(dataset.Dataset):
    """
    对Dataset重写
    """

    def __init__(self, buildData):
        super(myDataset, self).__init__()
        self.Data = buildData.Data
        self.length = buildData.length

    def __len__(self):
        assert self.length >= 0
        return self.length

    def __getitem__(self, item):
        inputs = {
            'data': self.Data[item][0],
            'label': self.Data[item][1]
        }
        return inputs
