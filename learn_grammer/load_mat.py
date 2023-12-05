import scipy
import numpy as np


def point_normalization(data):
    data_min = np.min(data, axis=-1)
    data_max = np.max(data, axis=-1)
    if data_min == data_max:
        return data
    else:
        return (data - data_min) / (data_max - data_min)


def load_data_from_file(filename):
    """
    load .mat file from input filename
    """
    mat_file = scipy.io.loadmat(filename)
    
    emg = mat_file['emg'] # shape(877073, 12)
    restimulus = mat_file['restimulus'] # shape(877072, 1)
    rerepetition = mat_file['rerepetition'] # shape(877072, 1)
    acc = mat_file['acc'] # shape(877073, 36)
    force = mat_file['force'] # shape(877073, 6)
    forcecal = mat_file['forcecal'] # shape(2, 6)
    activation = mat_file['activation'] # shape(877073, 6)

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
            # semg_point = point_normalization(emg[i])
            # acc_point = point_normalization(acc[i])
            semg_point = emg[i]
            acc_point = acc[i]
            movements[-1].append(semg_point.tolist()) # shape(*, 48)
    print("Data in {} loaded.".format(filename.replace('\\', '/')))
    return movements, labels


if __name__ == '__main__':
    filename = './dataset/DB2_E3/S1_E3_A1.mat'
    movements, labels = load_data_from_file(filename)