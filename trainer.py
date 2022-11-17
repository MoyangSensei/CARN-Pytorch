import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train # torch.utils.data.dataloader.DataLoader
        self.loader_test = loader.loader_test # torch.utils.data.dataloader.DataLoader
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step() # scheduler.step()
        epoch = self.optimizer.get_last_epoch() + 1 # num of epoch
        lr = self.optimizer.get_lr() # learning rate

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        # Two timers for data and model
        # The timer calls tic () in init to record the current time
        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        '''
                print(self.loader_train.__dict__)
                {'dataset': <data.MyConcatDataset object at 0x7f122e425040>, 
                'num_workers': 0, 
                'prefetch_factor': 2, 
                'pin_memory': True, 
                'pin_memory_device': '', 
                'timeout': 0, 
                'worker_init_fn': None, 
                '_DataLoader__multiprocessing_context': None, 
                '_dataset_kind': 0, 
                'batch_size': 16, 
                'drop_last': False, 
                'sampler': <torch.utils.data.sampler.RandomSampler object at 0x7f122e425130>, 
                'batch_sampler': <torch.utils.data.sampler.BatchSampler object at 0x7f122e4256a0>, 
                'generator': None, 
                'collate_fn': <function default_collate at 0x7f123055d430>, 
                'persistent_workers': False, 
                '_DataLoader__initialized': True, 
                '_IterableDataset_len_called': None, 
                '_iterator': None}
        '''
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            # size of lr: [bs, n_colors, patch_size/scale, patch_size/scale]  tensor
            # size of hr: [bs, n_colors, patch_size, patch_size] tensor
            lr, hr = self.prepare(lr, hr)
            # Data timer hold: call tic to calculate the distance from the current time to t0, and then accumulate: acc+tic()
            timer_data.hold()
            # The model timer records t0 again. The time of each round of model training starts here
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            # Back propagation, calculate the current gradient
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            # Update network parameters according to gradient
            self.optimizer.step()
            # The model has completed an epoch training here, accumulating time
            timer_model.hold()
            # get log file log.txt
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            # Reset t0 of data timer
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        # updata lr
        self.optimizer.schedule()

    def test(self):
        # It is prohibited to calculate and update gradients for testing data. The two can be used together to set a context for testing data, because it is testing while training
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            '''
                print(d.__dict__)
                {'dataset': <data.bcsr.BCSR object at 0x7fcaaf6d7610>, 'num_workers': 0, 'prefetch_factor': 2, 'pin_memory': True, 'pin_memory_device': '', 'timeout': 0, 'worker_init_fn': None, '_DataLoader__multiprocessing_context': None, '_dataset_kind': 0, 'batch_size': 1, 'drop_last': False, 'sampler': <torch.utils.data.sampler.SequentialSampler object at 0x7fcaaf6d7790>, 'batch_sampler': <torch.utils.data.sampler.BatchSampler object at 0x7fcaaf6db0d0>, 'generator': None, 'collate_fn': <function default_collate at 0x7fcab1815430>, 'persistent_workers': False, '_DataLoader__initialized': True, '_IterableDataset_len_called': None, '_iterator': None}
            '''
            # idx_data=0
            # d: dataloader
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    # size of lr: [batch_size=1, n_colors, input_size/scale, input_size/scale] tensor
                    # size of hr: [batch_size=1, n_colors, input_size, input_size] tensor
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        # Torch.device represents the object of the device to which torch.Tensor is assigned. torch. Device contains a device type ('cpu' or 'cuda') and an optional device serial number. If the equipment serial number does not exist, it is the current equipment.
        # For example, the result of torch. Tensor building 'cuda' with equipment is equivalent to 'cuda: X', where X is torch. cuda. current_ The result of device ().
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            # parser.add_argument('--precision', type=str, default='single',choices=('single', 'half'),help='FP precision for test (single | half)')
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

