import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if checkpoint.ok:
        # step 1 make dataloader
        loader = data.Data(args)
        # step 2 make network
        _model = model.Model(args, checkpoint)
        # step 3 loss function
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        # step 4 bring 123 into trainer
        t = Trainer(args, loader, _model, _loss, checkpoint)
        # step 5 train test run-step
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()

if __name__ == '__main__':
    main()
