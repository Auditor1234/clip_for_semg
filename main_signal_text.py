import clip
import torch
import argparse

from torch.optim import lr_scheduler, SGD, Adam
from dataloader import SignalWindow, h5py_to_window
from torch.utils.data import DataLoader
from train import train_one_epoch_signal_text, validate_signal_text, evaluate_signal_text
from utils import setup_seed, save_model_weight

import torch.nn as nn
import numpy as np


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
    model, _ = clip.load("RN50", device=device, vis_pretrain=False)
    
    # optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.2)
    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_func = nn.CrossEntropyLoss()

    filename = 'dataset/window.h5'
    weight_path = 'res/best.pt'
    best_precision, current_precision = 0, 0

    emg, label = h5py_to_window(filename)
    data_len = len(label)
    index = np.random.permutation(data_len)
    emg = emg[index] * 20000
    label = label[index]

    train_len = int(data_len * 0.8)
    val_len = int(data_len * 0.1)
    print('{} window for training, {} window for validation and {} data for test.'.format \
          (train_len, val_len, data_len - train_len - val_len))

    # 数据按照8:1:1分为训练集、验证集和测试集
    train_emg = emg[: train_len]
    train_label = label[: train_len]
    val_emg = emg[train_len : train_len + val_len]
    val_label = label[train_len : train_len + val_len]
    eval_emg = emg[train_len + val_len :]
    eval_label = label[train_len + val_len :]

    train_loader = DataLoader(
                    SignalWindow(train_emg, train_label),
                    batch_size=args.batch_size,
                    num_workers=0
                    )
    
    val_loader = DataLoader(
                    SignalWindow(val_emg, val_label),
                    batch_size=8,
                    num_workers=0
                    )

    eval_loader = DataLoader(
                    SignalWindow(eval_emg, eval_label),
                    batch_size=8,
                    num_workers=0
                    )


    model.train().half()
    model.to(device)
    print("start training...")
    for epoch in range(epochs):
        train_one_epoch_signal_text(model, epoch, epochs, device, train_loader, loss_func, optimizer, scheduler)
        current_precision = validate_signal_text(model, device, val_loader, loss_func)

        if current_precision > best_precision:
            best_precision = current_precision
            print('Current best precision in val set is:%.4f' % (best_precision * 100) + '%')
            save_model_weight(model=model, filename=weight_path)


    model.load_state_dict(torch.load(weight_path))
    evaluate_signal_text(model, device, eval_loader, loss_func)

if  __name__ == "__main__":
    args = arg_parse()
    main(args)
