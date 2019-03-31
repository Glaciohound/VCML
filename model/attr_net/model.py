import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models


PYTORCH_VER = torch.__version__


class AttributeNetwork():

    def __init__(self, opt):
        if opt.concat_img:
            self.input_channels = 6
        else:
            self.input_channels = 3

        if opt.load_checkpoint_path:
            print('| loading checkpoint from %s' % opt.load_checkpoint_path)
            checkpoint = torch.load(opt.load_checkpoint_path)
            if self.input_channels != checkpoint['input_channels']:
                raise ValueError('Incorrect input channels for loaded model')
            self.output_dim = checkpoint['output_dim']
            self.net = _Net(self.output_dim, self.input_channels)
            self.net.load_state_dict(checkpoint['model_state'])
        else:
            self.output_dim = (opt.dim_object, opt.dim_attribute, opt.dim_relation)
            self.net = _Net(self.output_dim, self.input_channels)

        self.criterion = nn.MSELoss(reduce='sum')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.learning_rate)

        self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.gpu_ids = opt.gpu_ids
        if self.use_cuda:
            self.net.cuda(opt.gpu_ids[0])

        self.input, self.label = None, None

    def set_input(self, x, label=None, relation=None):
        self.input = self._to_var(x)
        if label is not None:
            self.label = self._to_var(label)
            self.relation = self._to_var(relation)

    def step(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss.backward()
        self.optimizer.step()

    def forward(self):
        self.label_pred, self.features = self.net(self.input, get_feature=True)
        batch_size = self.label_pred.shape[0]
        self.loss = 0
        for i in range(batch_size):
            if self.label is not None:
                actual_objects = (self.label[i].sum(dim=1)!=0).sum()
                actual_objects = min(actual_objects+1, self.label.shape[1])
                self.loss += self.criterion(self.label_pred[i, :actual_objects],
                                            self.label[i, :actual_objects])
        return self.label_pred

    def get_loss(self):
        if PYTORCH_VER.startswith('0.4'):
            return self.loss.data.item()
        else:
            return self.loss.data

    def balanced_loss(self, pred, truth):
        n_True = (truth==1).sum()
        n_False = (truth==0).sum()
        loss_True = ((truth==1).float()*((pred-1).pow(2))).sum() / n_True if n_True != 0 else 0
        loss_False = ((truth==0).float()*(pred.pow(2))).sum() / n_False
        return (loss_True + loss_False)*(n_True + n_False) / 2

    def get_pred(self, get_feature=False):
        if not get_feature:
            return self.label_pred.data.cpu().numpy()
        else:
            return self.label_pred.data.cpu().numpy(),\
                self.features.data.cpu().numpy()

    def eval_mode(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def save_checkpoint(self, save_path):
        checkpoint = {
            'input_channels': self.input_channels,
            'output_dim': self.output_dim,
            'model_state': self.net.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if self.use_cuda:
            self.net.cuda(self.gpu_ids[0])


    def load_checkpoint(self, save_path):
        loaded = torch.load(save_path)
        self.net.cpu().load_state_dict(loaded)
        if self.use_cuda:
            self.net.cuda(self.gpu_ids[0])

    def _to_var(self, x):
        if self.use_cuda:
            x = x.cuda()
        return Variable(x)

class _Net(nn.Module):

    def __init__(self, output_dim, input_channels=6):
        super(_Net, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())

        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 6-channel input
        layers.pop(0)
        layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Linear(512, output_dim)

    def forward(self, x, get_feature=False):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        output = self.final_layer(x)
        if not get_feature:
            return output
        else:
            return output, x

def get_model(opt):
    model = AttributeNetwork(opt)
    return model
