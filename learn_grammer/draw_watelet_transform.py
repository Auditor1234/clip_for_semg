import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pywt

path = 'D:/Download/Browser/起风了抖音配乐_爱给网_aigei_com.wav'
y, sr = librosa.load(path, sr=16000)
wavename = 'morl'
totalscal = 4  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
cparam = 2 * fc * totalscal  # 常数c
scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
[cwtmatr, frequencies] = pywt.cwt(y, scales, wavename, 1.0 / sr)
t = np.arange(0, y.shape[0]/sr, 1.0/sr)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
plt.title = ("小波时频图")
plt.show()

