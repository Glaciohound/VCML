from argparse import ArgumentParser
import torch

class Config:
    def __init__(self):
        super(Config, self).__init__()
        args = self.parse_args()
        self.__dict__.update(vars(args))
        self.post_process()

    def parse_args(self):
        parser = ArgumentParser()

        parser.add_argument('--data_dir', default='data')
        parser.add_argument('--image_dir', default='data/raw/allImages/images')
        parser.add_argument('--sceneGraph_h5', default='data/processed/SG.h5')
        parser.add_argument('--sceneGraph_json', default='data/raw/sceneGraphs/all_sceneGraphs.json')
        parser.add_argument('--vocabulary_file', default='data/processed/vocabulary.json')
        parser.add_argument('--protocol_file', default='data/processed/protocol.json')
        parser.add_argument('--encodedQuestions_dir', default='data/processed/questions')

        parser.add_argument('--max_nImages', default=-1)
        parser.add_argument('--box_scale', default=1024)
        parser.add_argument('--image_scale', default=592)

        parser.add_argument('--num_workers', default=1)
        parser.add_argument('--train_shuffle', action='store_true')
        parser.add_argument('--train_batch_size', default=10)
        parser.add_argument('--val_batch_size', default=1)

        return parser.parse_args()

    def post_process(self):
        self.num_gpus = torch.cuda.device_count()
