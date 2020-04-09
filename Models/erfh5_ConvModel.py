import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d, Linear

from Models.model_utils import load_model_layers_from_path
from Utils.data_utils import reshape_to_indeces
from Utils.training_utils import count_parameters


class erfh5_Conv3d(nn.Module):
    def __init__(self, sequence_len):
        super(erfh5_Conv3d, self).__init__()
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv3d(1, 32, (17, 17, 17), padding=8)
        self.conv2 = nn.Conv3d(1, 64, (9, 9, 9), padding=4)
        self.conv3 = nn.Conv3d(64, 128, (5, 5, 5), padding=2)
        self.conv_f = nn.Conv3d(128, 1, (3, 3, 3), padding=1)

        self.conv_end = nn.Conv2d(sequence_len, 1, (3, 3), padding=1)

    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        # out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.dropout(out)
        out = F.relu(self.conv3(out))
        out = self.dropout(out)
        out = F.relu(self.conv_f(out))

        out = self.dropout(out)
        out = torch.squeeze(out, 1)

        out = self.conv_end(out)

        out = torch.squeeze(out, 1)

        return out


class SensorToDryspotBoolModel(nn.Module):
    def __init__(self):
        super(SensorToDryspotBoolModel, self).__init__()
        self.dropout = nn.dropout(0.1)
        self.maxpool = nn.maxpool2d(2, 2)
        self.conv1 = nn.conv2d(1, 32, (7, 7))
        self.conv2 = nn.conv2d(32, 64, (5, 5))
        self.conv3 = nn.conv2d(64, 128, (3, 3))
        self.conv4 = nn.conv2d(128, 256, (3, 3))

        self.fc1 = nn.linear(256, 1024)
        self.fc2 = nn.linear(1024, 512)
        self.fc3 = nn.linear(512, 128)
        self.fc_f = nn.linear(128, 1)

    def forward(self, x):
        out = x.reshape((-1, 1, 38, 30))
        out = self.dropout(out)
        out = F.relu(self.conv1(out))
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.dropout(out)
        out = F.relu(self.conv3(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out = F.relu(self.conv4(out))
        out = self.maxpool(out)
        out = self.dropout(out)

        out = out.view(out.size(0), 256, -1)

        out = out.sum(2)

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc_f(out)

        return out


class erfh5_Conv2dPercentage(nn.Module):
    def __init__(self):
        super(erfh5_Conv2dPercentage, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, (15, 15))
        self.conv2 = nn.Conv2d(32, 64, (7, 7))
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.conv4 = nn.Conv2d(128, 256, (3, 3))

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc_f = nn.Linear(128, 1)

    def forward(self, x):
        out = torch.unsqueeze(x, 1)
        out = self.dropout(out)
        out = F.relu(self.conv1(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out = F.relu(self.conv3(out))
        out = self.maxpool(out)
        out = self.dropout(out)
        out = F.relu(self.conv4(out))
        out = self.maxpool(out)
        out = self.dropout(out)

        out = out.view(out.size(0), 256, -1)

        out = out.sum(2)

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc_f(out)

        return out


class erfh5_Conv25D_Frame(nn.Module):
    def __init__(self, sequence_len):
        super(erfh5_Conv25D_Frame, self).__init__()
        self.conv1 = nn.Conv2d(sequence_len, 32, (15, 15), padding=7)
        self.conv2 = nn.Conv2d(32, 64, (7, 7), padding=3)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 1, (3, 3), padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.dropout(x)
        out = F.relu(self.conv1(out))

        out = self.dropout(out)
        out = F.relu(self.conv2(out))

        out = self.dropout(out)
        out = F.relu(self.conv3(out))

        out = self.dropout(out)
        out = F.relu(self.conv4(out))
        out = torch.squeeze(out, 1)

        return out


class DrySpotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 128, 13, stride=1, padding=0)
        self.conv2 = Conv2d(128, 256, 7, stride=1, padding=0)
        self.conv3 = Conv2d(256, 512, 5, stride=1, padding=0)
        self.conv4 = Conv2d(512, 1024, 3, padding=0)
        self.fc_f1 = nn.Linear(1024, 512)
        self.fc_f2 = nn.Linear(512, 256)
        self.fc_f3 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        a = x.reshape(-1, 1, 143, 111)
        b = F.relu(F.max_pool2d(self.conv1(a), kernel_size=2, stride=2))
        c = F.relu(F.max_pool2d(self.conv2(b), kernel_size=2, stride=2))
        d = F.relu(F.max_pool2d(self.conv3(c), kernel_size=2, stride=2))
        e = F.relu(self.conv4(d))
        f = e.view(e.shape[0], e.shape[1], -1).mean(2)
        f = self.dropout(f)
        g = F.relu(self.fc_f1(f))
        g = self.dropout(g)
        h = F.relu(self.fc_f2(g))
        h = self.dropout(h)
        i = torch.sigmoid(self.fc_f3(h))

        return i


class SensorDeconvToDryspot(nn.Module):
    def __init__(self, input_dim=1140):
        super(SensorDeconvToDryspot, self).__init__()
        self.fc = Linear(input_dim, 1140)

        self.ct1 = ConvTranspose2d(1, 16, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(16, 32, 7, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(32, 64, 15, stride=2, padding=0)
        self.ct4 = ConvTranspose2d(64, 64, 17, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.shaper0 = Conv2d(64, 32, 17, stride=2, padding=0)
        self.shaper = Conv2d(32, 64, 15, stride=2, padding=0)
        self.med = Conv2d(64, 128, 7, padding=0)
        self.details = Conv2d(128, 256, 3)
        self.details2 = Conv2d(256, 1024, 3, padding=0)

        self.linear2 = Linear(1024, 512)
        self.linear3 = Linear(512, 256)
        self.linear4 = Linear(256, 1)

    def forward(self, inputs):
        f = inputs
        # f = F.relu(self.fc(inputs))

        fr = f.reshape((-1, 1, 38, 30))
        fr = fr.contiguous()

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))
        k3 = F.relu(self.ct4(k3))

        t1 = F.relu(self.shaper0(k3))
        t1 = self.maxpool(t1)
        t1 = F.relu(self.shaper(t1))
        t1 = self.maxpool(t1)
        t2 = F.relu(self.med(t1))
        t2 = self.maxpool(t2)
        t3 = F.relu(self.details(t2))
        t3 = self.maxpool(t3)
        t4 = torch.sigmoid(self.details2(t3))
        v = t4.view((t4.shape[0], 1024, -1)).contiguous()
        out = v.mean(-1).contiguous()
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = F.relu(self.linear4(out))
        return out


class SensorDeconvToDryspot2(nn.Module):
    def __init__(self, pretrained=False, checkpoint_path=None, freeze_nlayers=0):
        super(SensorDeconvToDryspot2, self).__init__()
        self.ct1 = ConvTranspose2d(1, 16, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(16, 32, 7, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(32, 64, 15, stride=2, padding=0)
        self.ct4 = ConvTranspose2d(64, 128, 17, stride=2, padding=0)

        self.shaper0 = Conv2d(128, 64, 17, stride=2, padding=0)
        self.shaper = Conv2d(64, 32, 15, stride=2, padding=0)
        self.med = Conv2d(32, 32, 7, padding=0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear2 = Linear(1024, 512)
        self.linear3 = Linear(512, 256)
        self.linear4 = Linear(256, 1)
        if pretrained:
            self.load_model(checkpoint_path)

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            logger = logging.getLogger(__name__)
            logger.info(f'Freezing: {c}')

            for param in c.parameters():
                param.requires_grad = False
            if i == freeze_nlayers - 1:
                break

    def forward(self, inputs):
        f = inputs
        fr = f.reshape((-1, 1, 38, 30))

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))
        x = F.relu(self.ct4(k3))

        x = F.relu(self.shaper0(x))
        x = self.maxpool(x)
        x = F.relu(self.shaper(x))
        x = self.maxpool(x)
        x = F.relu(self.med(x))
        x = self.maxpool(x)
        x = x.view((x.shape[0], 1024, -1)).contiguous()
        x = x.mean(-1).contiguous()
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x

    def load_model(self, path):
        from collections import OrderedDict
        logger = logging.getLogger(__name__)
        logger.info(f'Loading model from {path}')
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')

        new_model_state_dict = OrderedDict()
        model_state_dict = checkpoint["model_state_dict"]
        names = {'ct1', 'ct2', 'ct3', 'ct4', 'shaper0'}
        for k, v in model_state_dict.items():
            splitted = k.split('.')
            name = splitted[1]  # remove `module.`
            if name in names:
                new_model_state_dict[f'{name}.{splitted[2]}'] = v
            else:
                continue
        self.load_state_dict(new_model_state_dict, strict=False)


class S80DeconvToDrySpotEff(nn.Module):
    def __init__(self, pretrained="", checkpoint_path=None, freeze_nlayers=0):  # Could be 7
        super(S80DeconvToDrySpotEff, self).__init__()
        self.ct1 = ConvTranspose2d(1, 128, 3, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(128, 64, 7, stride=2, padding=0)
        self.ct5 = ConvTranspose2d(64, 32, 15, stride=2, padding=0)
        self.ct6 = ConvTranspose2d(32, 8, 17, stride=2, padding=0)

        self.c1 = Conv2d(8, 32, 11, stride=2)
        self.ck = Conv2d(32, 32, 3, padding=0)
        self.cj = Conv2d(32, 1, 3, padding=0)

        self.cc2 = Conv2d(1, 16, 21)
        self.cc3 = Conv2d(16, 64, 13)
        self.cc4 = Conv2d(64, 256, 5)
        self.cc5 = Conv2d(256, 512, 3)
        self.cc6 = Conv2d(512, 1024, 1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(.3)
        self.lin1 = nn.Linear(1024 * 3, 512)
        self.lin3 = nn.Linear(512, 1)

        if pretrained == "deconv_weights":
            logger = logging.getLogger(__name__)
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={'ct1', 'ct3', 'ct5', 'ct6', 'c1', 'ck', 'cj'})
            incomp = self.load_state_dict(weights, strict=False)
            logger.debug(f'All layers: {self.state_dict().keys()}')
            logger.debug(f'Loaded weights but the following: {incomp}')

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            logger = logging.getLogger(__name__)
            logger.info(f'Freezing: {c}')

            for param in c.parameters():
                param.requires_grad = False
            if i == freeze_nlayers - 1:
                break

    def forward(self, inputs):
        inputs = inputs.reshape((-1, 1, 10, 8))
        x = F.relu(self.ct1(inputs))
        x = F.relu(self.ct3(x))
        x = F.relu(self.ct5(x))
        x = F.relu(self.ct6(x))

        x = F.relu(self.c1(x))
        x = F.relu(self.ck(x))
        x = F.relu(self.cj(x))
        ###
        x = F.relu(self.maxpool(self.cc2(x)))
        x = F.relu(self.maxpool(self.cc3(x)))
        x = F.relu(self.maxpool(self.cc4(x)))
        x = F.relu(self.cc5(x))
        x = F.relu(self.cc6(x))
        x = x.view((x.shape[0], 3 * 1024, -1)).contiguous()
        x = x.mean(-1).contiguous()
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        # x = F.relu(self.lin2(x))
        # x = self.dropout(x)
        x = torch.sigmoid(self.lin3(x))
        return x


class S80Deconv2ToDrySpotEff(nn.Module):
    def __init__(self, pretrained="", checkpoint_path=None,
                 freeze_nlayers=0,  # Could be 9
                 round_at: float = None,
                 demo_mode=False):
        super(S80Deconv2ToDrySpotEff, self).__init__()
        self.ct1 = ConvTranspose2d(1, 128, 3, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(128, 64, 7, stride=2, padding=0)
        self.ct5 = ConvTranspose2d(64, 32, 15, stride=2, padding=0)
        self.ct6 = ConvTranspose2d(32, 16, 17, stride=2, padding=0)
        self.ctr = ConvTranspose2d(16, 8, 19, stride=1, padding=0)

        self.c1 = Conv2d(8, 32, 11, stride=2, padding=1)
        self.cu = Conv2d(32, 64, 7, stride=1, padding=1)
        self.ck = Conv2d(64, 32, 3, padding=0)
        self.cj = Conv2d(32, 1, 3, padding=0)

        self.cc2 = Conv2d(1, 16, 21)
        self.cc3 = Conv2d(16, 64, 13)
        self.cc4 = Conv2d(64, 256, 5)
        self.cc5 = Conv2d(256, 512, 3)
        self.cc6 = Conv2d(512, 1024, 1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(.3)
        self.lin1 = nn.Linear(1024 * 2, 512)
        self.lin3 = nn.Linear(512, 1)

        self.round_at = round_at

        self.demo_mode = demo_mode

        if pretrained == "deconv_weights":
            logger = logging.getLogger(__name__)
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={'ct1', 'ct3', 'ct5', 'ct6', 'ctr',
                                                               'c1', 'cu', 'ck', 'cj'})
            incomp = self.load_state_dict(weights, strict=False)
            logger.debug(f'All layers: {self.state_dict().keys()}')
            logger.debug(f'Loaded weights but the following: {incomp}')
        elif pretrained == "all":
            logger = logging.getLogger(__name__)
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={'ct1', 'ct3', 'ct5', 'ct6', 'ctr',
                                                               'c1', 'cu', 'ck', 'cj',
                                                               'cc2', 'cc3', 'cc4', 'cc5', 'cc6',
                                                               'lin1', 'lin3'})
            incomp = self.load_state_dict(weights, strict=False)
            logger.debug(f'All layers: {self.state_dict().keys()}')
            logger.debug(f'Loaded weights but the following: {incomp}')

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            logger = logging.getLogger(__name__)
            logger.info(f'Freezing: {c}')

            for param in c.parameters():
                param.requires_grad = False
            if i == freeze_nlayers - 1:
                break

    def forward(self, inputs):
        if self.demo_mode:
            inputs = reshape_to_indeces(inputs, ((1, 4), (1, 4)), 80).contiguous()
        inputs = inputs.reshape((-1, 1, 10, 8))
        x = F.relu(self.ct1(inputs))
        x = F.relu(self.ct3(x))
        x = F.relu(self.ct5(x))
        x = F.relu(self.ct6(x))
        x = F.relu(self.ctr(x))

        x = F.relu(self.c1(x))
        x = F.relu(self.cu(x))
        x = F.relu(self.ck(x))
        x = F.relu(self.cj(x))
        ###
        if self.round_at is not None:
            x = x.masked_fill((x >= self.round_at), 1.)
            x = x.masked_fill((x < self.round_at), 0.)
        x = F.relu(self.maxpool(self.cc2(x)))
        x = F.relu(self.maxpool(self.cc3(x)))
        x = F.relu(self.maxpool(self.cc4(x)))

        x = F.relu(self.maxpool(self.cc5(x)))
        x = F.relu(self.cc6(x))
        x = x.view((x.shape[0], 2 * 1024, -1)).contiguous()
        x = x.mean(-1).contiguous()
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.lin3(x))
        return x


class S20DeconvToDrySpotEff(nn.Module):
    def __init__(self, pretrained="", checkpoint_path=None, freeze_nlayers=0):
        super(S20DeconvToDrySpotEff, self).__init__()
        self.ct1 = ConvTranspose2d(1, 256, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(256, 128, 5, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(128, 64, 10, stride=2, padding=0)
        self.ct4 = ConvTranspose2d(64, 16, 17, stride=2, padding=0)

        self.details = Conv2d(16, 8, 5)

        self.c2 = Conv2d(8, 16, 7, padding=0)
        self.c3 = Conv2d(16, 8, 5, padding=0)
        self.c4 = Conv2d(8, 1, 3, padding=0)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.lin1 = Linear(572, 512)
        self.lin2 = Linear(512, 256)
        self.lin3 = Linear(256, 1)

        if pretrained == "deconv_weights":
            logger = logging.getLogger(__name__)
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={'ct1', 'ct2', 'ct3', 'ct4',
                                                               'details'})
            incomp = self.load_state_dict(weights, strict=False)
            logger.debug(f'All layers: {self.state_dict().keys()}')
            logger.debug(f'Loaded weights but the following: {incomp}')

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            logger = logging.getLogger(__name__)
            logger.info(f'Freezing: {c}')

            for param in c.parameters():
                param.requires_grad = False
            if i == freeze_nlayers - 1:
                break

    def forward(self, inputs):
        frs = inputs.reshape((-1, 1, 5, 4))

        x = F.relu(self.ct1(frs))
        x = F.relu(self.ct2(x))
        x = F.relu(self.ct3(x))
        x = F.relu(self.ct4(x))

        x = F.relu(self.details(x))
        x = F.relu(self.c2(x))
        x = self.maxpool(x)
        x = F.relu(self.c3(x))
        x = self.maxpool(x)
        x = F.relu(self.c4(x))
        x = x.view((x.shape[0], 572, -1)).contiguous()
        x = x.mean(-1).contiguous()

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))

        return x


class S20DeconvToDrySpotEff2(nn.Module):
    def __init__(self, pretrained="", checkpoint_path=None, freeze_nlayers=0, round_at: float = None,
                 demo_mode=False):
        super(S20DeconvToDrySpotEff2, self).__init__()
        self.ct1 = ConvTranspose2d(1, 256, 3, stride=2)
        self.ct2 = ConvTranspose2d(256, 128, 5, stride=2)
        self.ct3 = ConvTranspose2d(128, 64, 10, stride=2)
        self.ct4 = ConvTranspose2d(64, 16, 17, stride=2)

        self.details = Conv2d(16, 8, 5)
        # ^ Pretrained ^
        self.c2 = Conv2d(8, 16, 13)
        self.c3 = Conv2d(16, 64, 7)
        self.c4 = Conv2d(64, 128, 3)
        self.c5 = Conv2d(128, 256, 3)
        self.c6 = Conv2d(256, 512, 3)
        self.c7 = Conv2d(512, 512, 1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.lin1 = Linear(1024, 256)
        self.lin2 = Linear(256, 1)

        self.dropout = nn.Dropout(0.3)
        # self.bn8 = nn.BatchNorm2d(8)
        # self.bn512 = nn.BatchNorm2d(512)
        self.round_at = round_at

        self.demo_mode = demo_mode

        if pretrained == "deconv_weights":
            logger = logging.getLogger(__name__)
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={'ct1', 'ct2', 'ct3', 'ct4',
                                                               'details'})
            incomp = self.load_state_dict(weights, strict=False)
            logger.debug(f'All layers: {self.state_dict().keys()}')
            logger.debug(f'Loaded weights but the following: {incomp}')
        if pretrained == "all":
            logger = logging.getLogger(__name__)
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={'ct1', 'ct2', 'ct3', 'ct4',
                                                               'details', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                                                               'lin1', 'lin2'})
            incomp = self.load_state_dict(weights, strict=False)
            logger.debug(f'All layers: {self.state_dict().keys()}')
            logger.debug(f'Loaded weights but the following: {incomp}')

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            logger = logging.getLogger(__name__)
            logger.info(f'Freezing: {c}')

            for param in c.parameters():
                param.requires_grad = False
            if i == freeze_nlayers - 1:
                break

    def forward(self, inputs):
        if self.demo_mode:
            inputs = reshape_to_indeces(inputs, ((1, 8), (1, 8)), 20).contiguous()
        frs = inputs.reshape((-1, 1, 5, 4))

        x = F.relu(self.ct1(frs))
        x = F.relu(self.ct2(x))
        x = F.relu(self.ct3(x))
        x = F.relu(self.ct4(x))
        x = F.relu(self.details(x))

        if self.round_at is not None:
            x = x.masked_fill((x >= self.round_at), 1.)
            x = x.masked_fill((x < self.round_at), 0.)

        # Shape: [1, 8, 127, 111]
        # x = self.bn8(x)
        x = F.relu(self.c2(x))
        x = self.maxpool(x)
        x = F.relu(self.c3(x))
        x = self.maxpool(x)
        x = F.relu(self.c4(x))
        x = self.maxpool(x)
        x = F.relu(self.c5(x))
        x = self.maxpool(x)
        x = F.relu(self.c6(x))
        x = F.relu(self.c7(x))

        # x = self.bn512(x)
        x = x.view((x.shape[0], 1024, -1)).contiguous()
        x = x.mean(-1).contiguous()
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.lin2(x))

        return x


class S20DeconvToDrySpotEff3(nn.Module):
    def __init__(self, pretrained="", checkpoint_path=None, freeze_nlayers=0):
        super(S20DeconvToDrySpotEff3, self).__init__()
        self.ct1 = ConvTranspose2d(1, 256, 3, stride=2)
        self.ct2 = ConvTranspose2d(256, 128, 5, stride=2)
        self.ct3 = ConvTranspose2d(128, 64, 10, stride=2)
        self.ct4 = ConvTranspose2d(64, 16, 17, stride=2)

        self.details = Conv2d(16, 8, 5)
        # ^ Pretrained ^
        self.c2 = Conv2d(8, 32, 13)
        self.c3 = Conv2d(32, 128, 7)
        self.c4 = Conv2d(128, 512, 3)
        self.c5 = Conv2d(512, 512, 3)
        self.c6 = Conv2d(512, 512, 1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.lin1 = Linear(1024, 256)
        self.lin2 = Linear(256, 1)

        self.dropout = nn.Dropout(0.3)
        self.bn8 = nn.BatchNorm2d(8)
        self.bn512 = nn.BatchNorm2d(512)

        if pretrained == "deconv_weights":
            logger = logging.getLogger(__name__)
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={'ct1', 'ct2', 'ct3', 'ct4',
                                                               'details'})
            incomp = self.load_state_dict(weights, strict=False)
            logger.debug(f'All layers: {self.state_dict().keys()}')
            logger.debug(f'Loaded weights but the following: {incomp}')

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            logger = logging.getLogger(__name__)
            logger.info(f'Freezing: {c}')

            for param in c.parameters():
                param.requires_grad = False
            if i == freeze_nlayers - 1:
                break

    def forward(self, inputs):
        frs = inputs.reshape((-1, 1, 5, 4))

        x = F.relu(self.ct1(frs))
        x = F.relu(self.ct2(x))
        x = F.relu(self.ct3(x))
        x = F.relu(self.ct4(x))
        x = F.relu(self.details(x))

        # Shape: [1, 8, 127, 111]
        x = self.bn8(x)
        x = F.relu(self.c2(x))
        x = self.maxpool(x)
        x = F.relu(self.c3(x))
        x = self.maxpool(x)
        x = F.relu(self.c4(x))
        x = self.maxpool(x)
        x = F.relu(self.c5(x))
        x = self.maxpool(x)
        x = F.relu(self.c6(x))

        x = self.bn512(x)
        x = x.view((x.shape[0], 1024, -1)).contiguous()
        x = x.mean(-1).contiguous()
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.lin2(x))

        return x


class SensorDeconvToDryspotEfficient(nn.Module):
    def __init__(self, pretrained="", checkpoint_path=None, freeze_nlayers=0):
        super(SensorDeconvToDryspotEfficient, self).__init__()
        self.ct1 = ConvTranspose2d(1, 128, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(128, 64, 7, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(64, 32, 15, stride=2, padding=0)
        self.ct4 = ConvTranspose2d(32, 8, 17, stride=2, padding=0)

        self.shaper0 = Conv2d(8, 16, 17, stride=2, padding=0)
        self.shaper = Conv2d(16, 32, 15, stride=2, padding=0)
        self.med = Conv2d(32, 32, 7, padding=0)
        self.details = Conv2d(32, 32, 3)
        ###
        self.details2 = Conv2d(32, 64, 13, padding=0)
        self.details3 = Conv2d(64, 128, 7, padding=0)
        self.details4 = Conv2d(128, 256, 5, padding=0)
        self.details5 = Conv2d(256, 512, 3, padding=0)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear2 = Linear(7680, 1024)
        self.linear3 = Linear(1024, 1)

        self.bn32 = nn.BatchNorm2d(32)
        self.bn512 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout(0.5)

        if pretrained == "deconv_weights":
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={"ct1", "ct2", "ct3", "ct4",
                                                               "shaper0", "shaper", "med", "details"})
            self.load_state_dict(weights, strict=False)
        elif pretrained == "all_weights":
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={"ct1", "ct2", "ct3", "ct4",
                                                               "shaper0", "shaper", "med", "details",
                                                               "details2", "details3", "details4", "details5",
                                                               "linear2", "linear3",
                                                               "bn32", "bn512"})
            self.load_state_dict(weights, strict=False)

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            logger = logging.getLogger(__name__)
            logger.info(f'Freezing: {c}')

            for param in c.parameters():
                param.requires_grad = False
            if i == freeze_nlayers - 1:
                break

    def forward(self, inputs):
        fr = inputs.reshape((-1, 1, 38, 30))

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))
        k3 = F.relu(self.ct4(k3))

        t1 = F.relu(self.shaper0(k3))
        t1 = F.relu(self.shaper(t1))
        t2 = F.relu(self.med(t1))
        t3 = F.relu(self.details(t2))

        # Shape: [1, 32, 151, 119]
        x = self.bn32(t3)
        x = F.relu(self.maxpool(self.details2(x)))
        x = F.relu(self.maxpool(self.details3(x)))
        x = F.relu(self.maxpool(self.details4(x)))
        x = F.relu(self.maxpool(self.details5(x)))
        x = self.bn512(x)
        x = x.view((x.shape[0], 7680, -1)).contiguous()
        x = x.mean(-1).contiguous()
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.linear3(x))
        return x


class SensorDeconvToDryspotEfficient2(nn.Module):
    def __init__(self, pretrained="", checkpoint_path=None, freeze_nlayers=0):
        super(SensorDeconvToDryspotEfficient2, self).__init__()
        self.ct1 = ConvTranspose2d(1, 128, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(128, 64, 7, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(64, 32, 15, stride=2, padding=0)
        self.ct4 = ConvTranspose2d(32, 8, 17, stride=2, padding=0)

        self.shaper0 = Conv2d(8, 16, 17, stride=2, padding=0)
        self.shaper = Conv2d(16, 32, 15, stride=2, padding=0)
        self.med = Conv2d(32, 32, 7, padding=0)
        self.details = Conv2d(32, 32, 3)
        ###
        self.details2 = Conv2d(32, 64, 13, padding=0)
        self.details3 = Conv2d(64, 128, 7, padding=0)
        self.details4 = Conv2d(128, 256, 5, padding=0)
        self.details5 = Conv2d(256, 512, 3, padding=0)
        self.details6 = Conv2d(512, 512, 3, padding=0)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear2 = Linear(1536, 1024)
        self.linear3 = Linear(1024, 1)

        self.bn32 = nn.BatchNorm2d(32)
        self.bn512 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout(0.3)

        if pretrained == "deconv_weights":
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={"ct1", "ct2", "ct3", "ct4",
                                                               "shaper0", "shaper", "med", "details"})
            self.load_state_dict(weights, strict=False)
        elif pretrained == "all_weights":
            weights = load_model_layers_from_path(path=checkpoint_path,
                                                  layer_names={"ct1", "ct2", "ct3", "ct4",
                                                               "shaper0", "shaper", "med", "details",
                                                               "details2", "details3", "details4",
                                                               "details5", "details6",
                                                               "linear2", "linear3",
                                                               "bn32", "bn512"})
            self.load_state_dict(weights, strict=False)

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            logger = logging.getLogger(__name__)
            logger.info(f'Freezing: {c}')

            for param in c.parameters():
                param.requires_grad = False
            if i == freeze_nlayers - 1:
                break

    def forward(self, inputs):
        fr = inputs.reshape((-1, 1, 38, 30))

        k = F.relu(self.ct1(fr))
        k2 = F.relu(self.ct2(k))
        k3 = F.relu(self.ct3(k2))
        k3 = F.relu(self.ct4(k3))

        t1 = F.relu(self.shaper0(k3))
        t1 = F.relu(self.shaper(t1))
        t2 = F.relu(self.med(t1))
        t3 = F.relu(self.details(t2))

        # Shape: [1, 32, 151, 119]
        x = self.bn32(t3)
        x = F.relu(self.maxpool(self.details2(x)))
        x = F.relu(self.maxpool(self.details3(x)))
        x = F.relu(self.maxpool(self.details4(x)))
        x = F.relu(self.maxpool(self.details5(x)))
        x = F.relu(self.details6(x))
        x = self.bn512(x)
        x = x.view((x.shape[0], 1536, -1)).contiguous()
        x = x.mean(-1).contiguous()
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.linear3(x))
        return x


class S20Channel4toDrySpot(nn.Module):
    def __init__(self):
        super(S20Channel4toDrySpot, self).__init__()
        self.c1 = Conv2d(4, 16, 3, stride=1)
        self.c2 = Conv2d(16, 64, 1, stride=1)
        self.c3 = Conv2d(64, 256, 1, stride=1)
        self.c4 = Conv2d(256, 512, 1, stride=1)
        self.c5 = Conv2d(512, 1024, 1, stride=1)
        self.c6 = Conv2d(1024, 2048, 1, stride=1)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, _inputs):
        _inputs = _inputs.contiguous()
        x = _inputs.permute(0, 2, 1).reshape((-1, 4, 5, 4))
        x = x.contiguous()
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))
        x = F.relu(self.c6(x)).contiguous()

        x = x.view((x.shape[0], 2048, -1)).contiguous()
        x = x.mean(-1).contiguous()

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x


if __name__ == "__main__":
    model = S80Deconv2ToDrySpotEff(freeze_nlayers=9)
    print('param count:', count_parameters(model))
    m = model.cuda()
    em = torch.randn((1, 80)).cuda()
    out = m(em)

    print('end', out.shape)

    # # torch.tensor(np.arange(1., 1141.)).reshape((38, 30))[1::8, 1::8]
    # # Look up in PAM RTM or plot
    # # for 1::8, 1::8
    # # tensor([[  32.,   40.,   48.,   56.],
    # #         [ 272.,  280.,  288.,  296.],
    # #         [ 512.,  520.,  528.,  536.],
    # #         [ 752.,  760.,  768.,  776.],
    # #         [ 992., 1000., 1008., 1016.]], dtype=torch.float64)
    # out = m(em)
    #
    # print(out.shape)
