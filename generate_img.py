import os
import glob
import pywt
import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_emg_label_from_file, load_data_from_file


def draw_specgram(movement, img_path):
    Fs = 2000 # Fs：采样频率，默认为2
    NFFT = Fs # FFT中每个片段的数据点数（窗长度）。默认为256
    noverlap = int(1.0 / 3 * NFFT) # noverlap：窗之间的重叠长度。默认值是128。
    movement = np.mean(movement, axis=-1)
    # plt.plot(movement)

    plt.specgram(movement, NFFT=NFFT, Fs=Fs, noverlap=noverlap, 
                 mode='default', scale_by_freq=True, sides='default', scale='dB', xextent=None, vmin=-190, vmax=-90) # 统一刻度，设置最大最小值

    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight')
    return img_path


def draw_specgram_from_channel(movement, img_path):
    Fs = 2000 # Fs：采样频率，默认为2
    NFFT = Fs # FFT中每个片段的数据点数（窗长度）。默认为256
    noverlap = int(1.0 / 3 * NFFT) # noverlap：窗之间的重叠长度。默认值是128。
    channels = len(movement)

    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        plt.specgram(movement[i], NFFT=NFFT, Fs=Fs, noverlap=noverlap, 
                     mode='default', scale_by_freq=True, sides='default', scale='dB', xextent=None, vmin=-190, vmax=-90) # 统一刻度，设置最大最小值
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    plt.savefig(img_path, bbox_inches='tight')
    return img_path


def generate_specgram_from_path(path):
    files = glob.glob(f"{path}/*/*.mat")
    for file in files:
        movements, labels = load_data_from_file(file)
        dir_file = file.rstrip('.mat').split('\\')
        img_path = path + '/' + 'img/' + '_'.join(dir_file[2:]) + '_%d_%d.png'
        for i in range(len(movements)):
            draw_specgram(movements[i], img_path % (i, labels[i]))


def generate_specgram_channels_from_path(path):
    files = glob.glob(f"{path}/*/*.mat")
    path += '/channel_img/'
    if not os.path.exists(path):
        os.makedirs(path)
    for file in files:
        movements, labels = load_data_from_file(file)
        dir_file = file.rstrip('.mat').split('\\')
        img_path = path + '_'.join(dir_file[2:]) + '_%d_%d.png'
        for i in range(len(movements)):
            draw_specgram_from_channel(np.array(movements[i]).T, img_path % (i, labels[i]))


def draw_cwt(window_data, img_path):
    fs = 2000
    wavename = 'cgau8'
    totalscal = 20  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)

    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    [cwtmatr, frequencies] = pywt.cwt(window_data, scales, wavename, 1.0 / fs)
    t = np.arange(0, window_data.shape[0]/fs, 1.0/fs)

    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.axis('off')
    plt.subplots_adjust(hspace=0.4)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
    plt.savefig(img_path, bbox_inches = 'tight')


def generate_cwt_from_path(filename, img_dir):
    window_size = 200
    window_overlap = 0
    window_step = window_size - window_overlap
    assert window_step > 0, print("Window size must bigger than window overlap!")

    movements, labels = load_data_from_file(filename)
    exact_file = os.path.basename(filename).split('.')[0]
    img_count = 0
    for i, movement in enumerate(movements):
        if labels[i] == 0:
            continue
        movement_len = len(movement)
        movement = np.array(movement)
        movement = np.mean(movement, axis=-1)
        for index in range(0, movement_len, window_step):
            img_path = img_dir + '/' + exact_file + '_' + str(index // window_step) + '_' + str(labels[i]) + '.png'
            draw_cwt(movement[index : index + window_size], img_path)
            img_count += 1
            print('{} image has been drawn.'.format(img_count))


def draw_cwt_from_channel(window_data, img_path):
    fs = 2000
    wavename = 'morl'
    totalscal = 256  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）

    channels = 8
    window_data = np.array(window_data) * 20000
    for i in range(channels):
        [cwtmatr, frequencies] = pywt.cwt(window_data[:, i], scales, wavename, 1.0 / fs)
        t = np.arange(0, window_data.shape[0]/fs, 1.0/fs)
        plt.subplot(channels, 1, i + 1)
        plt.contourf(t, frequencies, abs(cwtmatr))
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig(img_path, bbox_inches='tight')

def draw_cwt_plot(window_data, img_path):
    fs = 2000
    wavename = 'morl'
    totalscal = 256  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）

    channels = 8
    window_data = np.array(window_data) * 20000
    [cwtmatr, frequencies] = pywt.cwt(window_data[:, 0], scales, wavename, 1.0 / fs)
    t = np.arange(0, window_data.shape[0]/fs, 1.0/fs)
    plt.contourf(t, frequencies, abs(cwtmatr), levels=np.linspace(0,10,41))
    plt.colorbar()
    plt.show()

def generate_cwt_from_channel(filename, img_dir):
    window_size = 400
    window_overlap = 0
    window_step = window_size - window_overlap
    assert window_step > 0, print("Window size must bigger than window overlap!")

    emg, label = load_emg_label_from_file(filename)
    exact_file = os.path.basename(filename).split('.')[0]
    
    for i in range(6, len(label)):
        movement_points = emg[i]
        movement_len = len(movement_points)
        print('Type {} has {} points, window size is {}.'.format(label[i], movement_len, window_size))
        img_count = 0
        for index in range(0, movement_len, window_step):
            img_path = img_dir + '/' + exact_file + '_' + str(index // window_step) + '_' + str(label[i]) + '.png'
            draw_cwt_from_channel(movement_points[index : index + window_size], img_path)
            img_count += 1
        print('Type {}, {} image has been drawn.\n'.format(label[i], img_count))



if __name__ == "__main__":
    filename = 'D:/Download/Datasets/Ninapro/DB2/S1/S1_E1_A1.mat'
    img_dir = 'dataset/img'
    generate_cwt_from_channel(filename, img_dir)