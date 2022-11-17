import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = False
        self.self_ensemble = False
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            forward_function = self.model.forward
            return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)