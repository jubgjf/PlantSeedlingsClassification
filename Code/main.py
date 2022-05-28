import os
from argparser import Parser
from config import config
from dataset import getDataset
from trainer import trainer
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = Parser()
    print(parser.args)
    config = config(parser.args)
    trainer = trainer(config)
    trainer.train()
    trainer.test()
