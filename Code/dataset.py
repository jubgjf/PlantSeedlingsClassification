import os
import torch.utils.data.dataset as dataset
from PIL import Image
from torchvision.transforms import transforms
from transformers import AutoFeatureExtractor


def getDataset(config):
    """
    获取 train + dev + test 的 dataset

    Args:
        config: 模型配置

    Returns:
        一个 tuple，内部的元素是
        (trainDataset, devDataset, testDataset), (label_str2int, label_int2str)
        其中 *Dataset 是数据迭代器
        label_str2int 和 label_int2str 是标签和 int 的相互转换的字典
    """

    trainBuildData = buildData(config, "train")
    devBuildData = buildData(config, "dev")
    testBuildData = buildData(config, "test")

    isTransformer = False
    if config.model == "SwinTransformer":
        isTransformer = True

    trainDataset = myDataset(trainBuildData, "train_dev", isTransformer)
    devDataset = myDataset(devBuildData, "train_dev", isTransformer)
    testDataset = myDataset(testBuildData, "test", isTransformer)

    return (trainDataset, devDataset, testDataset), (trainBuildData.label_str2int, trainBuildData.label_int2str)


class buildData():
    def __init__(self, config, choice='train'):
        """
        读入数据，并数据增强，生成一个数据例

        Args:
            config: 模型配置
            choice: 数据集类型，只能是 "train", "dev", "test" 之一
        """

        self.Data = []
        self.length = 0

        # 双向索引 label
        self.label_int2str = {}  # { 0: "Black-grass", 1: "Charlock", ...}
        self.label_str2int = {}  # { "Black-grass": 0, "Charlock": 1, ...}

        path_label = []

        # ===== train/dev =====
        path = config.train_path
        path_label_tmp = []
        for label_int, label_dir in enumerate(os.listdir(path)):
            self.label_int2str[label_int] = label_dir
            self.label_str2int[label_dir] = label_int
            if choice == 'train' or choice == 'dev':
                image_names = os.listdir(path + label_dir)
                for image_name in image_names:
                    path_label_tmp.append((path + label_dir + "/" + image_name, label_int))
        if choice == 'train' or choice == 'dev':
            for index, item in enumerate(path_label_tmp):
                if choice == 'train' and index % (config.train_dev_frac + 1) < config.train_dev_frac:
                    path_label.append(item)
                elif choice == 'dev' and index % (config.train_dev_frac + 1) >= config.train_dev_frac:
                    path_label.append(item)

        # ===== test =====
        elif choice == 'test':
            path = config.test_path
            image_names = os.listdir(path)
            for image_name in image_names:
                path_label.append((path + image_name, image_name))

        # 根据 config 进行数据增强
        trans_with_aug, trans_no_aug = self.dataAugment(config)

        # 当 choice 为 "train" 或 "dev" 时：Data 为 list, list 中每一项为二元组 (tensor, label)
        # 当 choice 为 "test" 时：Data 为 list, list 中每一项为二元组 (tensor, image_name)
        for data in path_label:
            self.Data.append((trans_with_aug(Image.open(data[0]).convert('RGB')), data[1]))
            self.Data.append((trans_no_aug(Image.open(data[0]).convert('RGB')), data[1]))
        self.length = len(self.Data)

    def dataAugment(self, config):
        """
        进行数据增强

        Args:
            config: 模型配置

        Returns:
            返回 transforms.Compose
        """

        trans_with_aug = []  # 有增强的 transform
        trans_no_aug = []  # 没有增强的 transform

        if config.data_augmentation == 'None':
            pass
        else:
            if 'rot' in config.data_augmentation:
                # TODO 旋转会导致图片出现大量的 0，不知道为什么，需要修一下
                trans_with_aug.append(transforms.RandomRotation(degrees=(0, 180)))  # 随机旋转
            if 'flp' in config.data_augmentation:
                trans_with_aug.append(transforms.RandomHorizontalFlip())  # 随机水平翻转
                trans_with_aug.append(transforms.RandomVerticalFlip())  # 随机垂直翻转
            if 'pst' in config.data_augmentation:
                trans_with_aug.append(transforms.RandomPosterize(bits=2))  # 随机分色
            if 'slt' in config.data_augmentation:
                trans_with_aug.append(transforms.RandomSolarize(threshold=192.0))  # 随机曝光
        trans_with_aug.append(transforms.Resize((224, 224)))  # 缩放到 224x224
        trans_with_aug.append(transforms.ToTensor())

        trans_no_aug.append(transforms.Resize((224, 224)))  # 缩放到 224x224
        trans_no_aug.append(transforms.ToTensor())

        return transforms.Compose(trans_with_aug), transforms.Compose(trans_no_aug)


class myDataset(dataset.Dataset):
    """
    对Dataset重写
    """

    def __init__(self, buildData, choice: str, isTransformer: bool):
        """
        Args:
            buildData: buildData 的一个实例
            choice: 只能取值为 "train_dev" 或 "test"
            isTransformer: 模型是否是 SwinTransformer
        """

        super(myDataset, self).__init__()
        self.choice = choice
        self.isTransformer = isTransformer
        self.Data = buildData.Data
        self.length = buildData.length
        if self.isTransformer:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    def __len__(self):
        assert self.length >= 0
        return self.length

    def __getitem__(self, item):
        data = self.Data[item][0]
        if self.isTransformer:
            data = self.feature_extractor(self.Data[item][0], return_tensors="pt").data['pixel_values'].squeeze(0)

        if self.choice == 'train_dev':
            inputs = {
                'data': data,
                'label': self.Data[item][1]
            }
        elif self.choice == 'test':
            inputs = {
                'data': data,
                'name': self.Data[item][1]
            }
        else:
            # 不应该到这个分支
            inputs = {
                'data': data,
                'label': self.Data[item][1]
            }
        return inputs
