from argparse import ArgumentParser, Namespace

class Config(Namespace):
    def __init__(self):
        super(Config, self).__init__()
        args = self.parse_args()
        self.__dict__.update(vars(args))

    def parse_args(self):
        parser = ArgumentParser()

        parser.add_argument('--data_dir', default='data')
        parser.add_argument('--image_dir', default='allImages/images')

        return parser.parse_args()
