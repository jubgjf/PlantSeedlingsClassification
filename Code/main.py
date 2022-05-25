from argparser import Parser
from config import config
from dataset import getDataset

if __name__ == '__main__':
    parser = Parser()
    print(parser.args)

    config = config(parser.args)

    trainDataset, devDataset, testDataset = getDataset(config)
    for index, data in enumerate(trainDataset):
        pass
