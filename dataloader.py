import torch
import scipy.io
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py


class SignalTo2D(Dataset):
    def __init__(self, signals, labels) -> None:
        self.signals = signals
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

    @staticmethod
    def collate_fn(train_data):
        signals, labels = [], []
        for signal, label in train_data:
            signals.append(torch.tensor(signal))
            labels.append(label)
        # pad with zero and get equal length tensor
        signals = pad_sequence(signals, batch_first=True)
        return signals, labels


class SignalSpecgram(Dataset):
    def __init__(self, path, data_type='train') -> None:
        super().__init__()
        self.path = path
        self.files = glob.glob(f"{path}/{data_type}/*.png")
        if len(self.files) == 0:
            print("No image file found!")
            exit(1)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self.files[idx]
    
    @staticmethod
    def collate_fn(train_data):
        labels = []
        for filename in train_data:
            labels.append(int(filename.rstrip('.png').split('_')[-1]))
        return train_data, labels


class SignalWindow(Dataset):
    def __init__(self, window_data, window_label) -> None:
        super().__init__()
        self.window_data = window_data
        self.window_label = window_label

    def __len__(self):
        return len(self.window_label)
    
    def __getitem__(self, idx):
        return self.window_data[idx, :, :8], self.window_label[idx]
    
    # @staticmethod
    # def collate_fn(train_data):
    #     labels = []
    #     for filename in train_data:
    #         labels.append(int(filename.rstrip('.png').split('_')[-1]))
    #     return train_data, labels


def load_data_from_file(filename):
    """
    load .mat file from input filename
    """
    mat_file = scipy.io.loadmat(filename)
    
    emg = mat_file['emg'] # shape(877073, 12)
    restimulus = mat_file['restimulus'] # shape(877072, 1)
    rerepetition = mat_file['rerepetition'] # shape(877072, 1)
    # acc = mat_file['acc'] # shape(877073, 36)
    # force = mat_file['force'] # shape(877073, 6)
    # forcecal = mat_file['forcecal'] # shape(2, 6)
    # activation = mat_file['activation'] # shape(877073, 6)

    movements = []
    labels = []

    print("{} data points in file {} found.".format(len(emg), filename.replace('\\', '/')))
    for i in range(len(emg) - 1):
        if rerepetition[i] < 1:
            continue
        if i == 0 or restimulus[i] != restimulus[i - 1]:
            movements.append([])
            labels.append(restimulus[i][0])
        else:
            movements[-1].append(emg[i].tolist()) # shape(*, 48)
    print("Data in {} loaded.".format(filename.replace('\\', '/')))
    return movements, labels


def load_data_from_dir(path):
    """
    load .mat file from input directory path
    """
    files = glob.glob(f"{path}/*/*.mat")
    if len(files) == 0:
        print("Error: no dataset found!")
        exit(1)

    total_movements, total_labels = [], []
    print("Loading data...")
    for file in files:
        movements, labels = load_data_from_file(file)
        total_movements += movements
        total_labels += labels
    print("{} movements have been loaded.".format(len(total_labels)))
    return total_movements, total_labels


def preprocess_sepcgram(filenames: torch.Tensor, preprocess):
    images = []
    for file in filenames:
        images.append(preprocess(Image.open(file)))
    return torch.stack(images, dim=0)


def load_emg_label_from_file(filename, class_type=10):
    emg, label = [], []
    for i in range(class_type):
        emg.append([])

    # iterate each file
    mat_file = scipy.io.loadmat(filename)
    file_emg = mat_file['emg']
    file_label = mat_file['restimulus']


    # store one file data except 'rest' action
    for i in range(len(file_label)):
        label_idx = file_label[i][0]
        if label_idx == 0 or label_idx > class_type:
            continue
        movement_idx = label_idx - 1
        if len(emg[movement_idx]) == 0:
            label.append(label_idx)
        emg[movement_idx].append(file_emg[i].tolist())
    print('{} has read, get {} types movement.'.format(filename, class_type))


    print('emg.length = ', len(emg))
    print('label = \n', label)

    return emg, label


def window_to_h5py(emg, label, filename, window_size=400, window_overlap=0):
    window_data = []
    window_label = []
    for i in range(len(label)):
        emg_type = np.array(emg[i])
        window_count = 0
        print('{} emg points found in type {} emg signal.'.format(len(emg_type), label[i]))
        for j in range(0, len(emg_type) - window_size, window_size - window_overlap):
            window_data.append(emg_type[j : j + window_size])
            window_label.append(label[i])
            window_count += 1
        print('{} window data found in type {} emg signal.'.format(window_count, label[i]))
    
    file = h5py.File(filename,'w')  
    file.create_dataset('windowData', data = np.stack(window_data, axis=0))
    file.create_dataset('windowLabel', data = np.array(window_label))
    file.close()


def h5py_to_window(filename):
    file = h5py.File(filename, 'r')
    emg = file['windowData'][:]
    label = file['windowLabel'][:]
    file.close()
    return emg, label


if __name__ == '__main__':
    filename = 'D:/Download/Datasets/Ninapro/DB2/S1/S1_E1_A1.mat'
    h5_filename = 'dataset/window_400_200.h5'
    emg, label = load_emg_label_from_file(filename)
    window_to_h5py(emg, label, h5_filename, window_overlap=200)
    emg, label = h5py_to_window(h5_filename)
    print(emg.shape)
    print(label.shape)