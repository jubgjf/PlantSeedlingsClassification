import torch.utils.data.dataset as dataset

class buildData():
    """
    TODO:读入数据，并数据增强，返回一个数据例Choice为train时返回训练数据，buildData.Data为List,List中每一项为二元组，buildData.length为长度int
    """
    def __init__(self, config, choice='train'):
        """
        :param config:config
        :param choice: train、dev、test，当使用数据增强时，使用不同的数据增强
        """
        pass

    def dataAugment(self):
        """
        数据增强
        """
        pass

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