import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d, Linear


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

        # self.conv1 = Conv2d(1, 128, 13, stride=1, padding=0)
        # self.conv2 = Conv2d(128, 512, 7, stride=3, padding=0)
        # self.conv3 = Conv2d(512, 2048, 5, stride=3, padding=0)
        # self.conv4 = Conv2d(2048, 4096, 3, padding=0)
        # self.fc_f1 = nn.Linear(4096, 1028)
        # self.fc_f2 = nn.Linear(1028, 512)
        # self.fc_f3 = nn.Linear(512, 1)
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

        self.linear2 = Linear(1024,512)
        self.linear3 = Linear(512,256)
        self.linear4 = Linear(256,1)

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
            print(f'Freezing: {c}')

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


class SensorDeconvToDryspotEfficient(nn.Module):
    def __init__(self, pretrained=False, checkpoint_path=None, freeze_nlayers=8):
        super(SensorDeconvToDryspotEfficient, self).__init__()
        self.ct1 = ConvTranspose2d(1, 128, 3, stride=2, padding=0)
        self.ct2 = ConvTranspose2d(128,64, 7, stride=2, padding=0)
        self.ct3 = ConvTranspose2d(64, 32, 15, stride=2, padding=0)
        self.ct4 = ConvTranspose2d(32, 8, 17, stride=2, padding=0)

        self.shaper0    = Conv2d(8, 16, 17, stride=2, padding=0)
        self.shaper     = Conv2d(16, 32, 15, stride=2, padding=0)
        self.med        = Conv2d(32, 32, 7, padding=0)
        self.details    = Conv2d(32, 32, 3)
        self.details2   = Conv2d(32, 16, 3, padding=0)
        self.details3   = Conv2d(16, 1, 3, padding=0)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear2 = Linear(884, 512)
        self.linear3 = Linear(512, 256)
        self.linear4 = Linear(256, 1)

        if pretrained:
            self.load_model(checkpoint_path)

        if freeze_nlayers == 0:
            return

        for i, c in enumerate(self.children()):
            print(f'Freezing: {c}')

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

        x = self.maxpool(t3)
        x = F.relu(self.details2(x))
        x = self.maxpool(x)
        x = F.relu(self.details3(x))
        x = x.view((x.shape[0], 884, -1)).contiguous()
        x = x.mean(-1).contiguous()
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x

    def load_model(self, path):
        logger = logging.getLogger(__name__)
        logger.info(f'Loading model from {path}')
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')

        new_model_state_dict = OrderedDict()
        model_state_dict = checkpoint["model_state_dict"]
        names = {'ct1', 'ct2', 'ct3', 'ct4', 'shaper0', 'shaper', 'med', 'details'}
        for k, v in model_state_dict.items():
            splitted = k.split('.')
            name = splitted[1]  # remove `module.`
            if name in names:
                new_model_state_dict[f'{name}.{splitted[2]}'] = v
            else:
                continue
        self.load_state_dict(new_model_state_dict, strict=False)


if __name__ == "__main__":
    model = SensorDeconvToDryspot()
    # model = SensorDeconvToDryspotEfficient()
    m = model.cuda()
    em = torch.empty((1, 1140)).cuda()
    out = m(em)

    print(out.shape)
