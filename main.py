import clip
import torch
import argparse

from torch.optim import lr_scheduler, SGD, Adam
from dataloader import SignalSpecgram 
from torch.utils.data import DataLoader
from train import train_one_epoch, evaluate, train_one_cnn_epoch, evaluate_cnn
from utils import setup_seed

import torch.nn as nn


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="dataset batch size")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--dataset", type=str, default="./dataset/img", help="dataset directory path")

    return parser.parse_args()

def main(args):
    setup_seed()
    
    epochs = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device, vis_pretrain=False)
    
    # optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.2)
    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_func = nn.CrossEntropyLoss()

    train_loader = DataLoader(
                    SignalSpecgram(args.dataset, data_type='train'),
                    batch_size=args.batch_size,
                    collate_fn=SignalSpecgram.collate_fn,
                    num_workers=0
                    )

    eval_loader = DataLoader(
                    SignalSpecgram(args.dataset, data_type='eval'),
                    batch_size=8,
                    collate_fn=SignalSpecgram.collate_fn,
                    num_workers=0
                    )

    model.train().half()
    model.to(device)
    print("start training...")
    for epoch in range(epochs):
        train_one_cnn_epoch(model, epoch, epochs, device, train_loader, loss_func, optimizer, preprocess, scheduler)
        if (epoch + 1) % 3 == 0:
            with torch.no_grad():
                evaluate_cnn(model, device, eval_loader, loss_func, preprocess)
        # for name, parms in model.named_parameters():
	    #     print('-->name:', name,  ' -->grad_value:', 'None' if parms.grad is None else torch.mean(parms.grad))
        

if  __name__ == "__main__":
    args = arg_parse()
    main(args)
