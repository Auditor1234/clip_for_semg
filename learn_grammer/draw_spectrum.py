import matplotlib.pyplot as plt
import numpy as np
from load_mat import load_data_from_file

def _normalize(data):
    data_min = np.min(data, axis=0, keepdims=True)
    data_max = np.max(data, axis=0, keepdims=True)
    return (data - data_min) / (data_max - data_min)



filename = 'dataset/DB2_E3/S1_E3_A1.mat'
movements, labels = load_data_from_file(filename)
index = 2 # 0-8, 动作的编号
rest = 1 # 是否是休息

label = labels[index * 12 + rest]
movement = np.array(movements[index * 12 + rest]).T

def draw_signal(movement):
    plt.plot(movement[0])
    plt.show()

def draw_aggregate(movement, NFFT, Fs, noverlap):
    channels = len(movement)
    # channels = 4

    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        plt.specgram(movement[i], NFFT=NFFT, Fs=Fs, noverlap=noverlap, 
                     mode='default', scale_by_freq=True, sides='default', scale='dB', xextent=None, vmin=-190, vmax=-90) # 统一刻度，设置最大最小值
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.show()


def draw_single(movement, NFFT, Fs, noverlap):
    plt.specgram(movement[0], NFFT=NFFT, Fs=Fs, noverlap=noverlap, 
             mode='default', scale_by_freq=True, sides='default', scale='dB', xextent=None) # 统一刻度，设置最大最小值
    fig_path = 'dataset/img/spectrum_1_0.png'
    plt.colorbar()
    plt.title('specgram')
    plt.ylabel('amplitude')
    plt.xlabel('time')
    plt.savefig(fig_path, bbox_inches = 'tight')



Fs = 2000 # Fs：采样频率，默认为2
NFFT = Fs # FFT中每个片段的数据点数（窗长度）。默认为256
noverlap = int(1.0 / 3 * NFFT) # noverlap：窗之间的重叠长度。默认值是128。

draw_single(movement, NFFT, Fs, 0)
