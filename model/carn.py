from model import common
import torch.nn as nn
from torch.nn import functional as F
import torch

def make_model(args, parent=False):
    return CARN(args)

class CAR(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAR, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down_and_up = nn.Sequential(
                # before down conv
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                # down conv
                nn.Conv2d(channel // reduction, channel // reduction, 1, padding=0, bias=True),
                # first activation
                nn.ReLU(inplace=True),
                # before up conv
                nn.Conv2d(channel // reduction, channel // reduction, 1, padding=0, bias=True),
                # up conv
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                # second activation
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.down_and_up(y)
        return x * y

class CARB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(CARB, self).__init__()
        modules_body = []
        for i in range(4):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i==1 or i==3: modules_body.append(act)
        modules_body.append(CAR(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class CARN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CARN, self).__init__()
        
        n_CARB = args.n_resgroups # num of carb
        n_feats = args.n_feats # num of feat. ch.
        kernel_size = 3 # kernel size of conv.
        reduction = args.reduction # reduction
        scale = args.scale[0] # upsample scale [2,3,4,8]
        act = nn.ReLU(True) # relu layer in res. block
        
        # 0.1 RGB mean for BCSR
        self.sub_mean = common.MeanShift(args.rgb_range) # sign default = -1
        
        # 1 define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # 2 define body module
        modules_body=[]
        for i in range(n_CARB):
            modules_body.append(
                CARB(conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # 3 define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        # 0.2 return submean result
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # ovalall: 1-2-3
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x # long skip
        x = self.tail(res)
        x = self.add_mean(x)
        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

