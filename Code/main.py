from argparser import Parser
from config import config

if __name__ == '__main__':
    parser = Parser()
    print(parser.args)

    config = config(parser.args)
