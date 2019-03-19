import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)
from config import Config
from dataset.dataset import DataLoader#, Dataset

args = Config()
train, val, test = DataLoader.get_dataloaders(args)
from IPython import embed; embed()
