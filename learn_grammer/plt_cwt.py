import matplotlib.pyplot as plt
import numpy as np
import pywt
from load_mat import load_data_from_file


filename = 'dataset/DB2_E3/S1_E3_A1.mat'
movements, labels = load_data_from_file(filename)

movement = np.array(movements[9]).T
# plt.plot(movement)
# plt.savefig('my_fig.png')
# plt.show()


sr = 2000
y = movement[0][:400]
plt.plot(movement[:8].T[:400])
plt.show()
wavename = 'morl'
totalscal = 256 # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
cparam = 2 * fc * totalscal  # 常数c
scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
[cwtmatr, frequencies] = pywt.cwt(y, scales, wavename, 1.0 / sr)
t = np.arange(0, y.shape[0]/sr, 1.0/sr)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.title("CWT")
plt.ylabel("freq")
plt.xlabel("time")
plt.subplots_adjust(hspace=0.4)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
plt.colorbar()
fig_path = 'dataset/img/cwt_1_0_part.png'
plt.savefig(fig_path, bbox_inches = 'tight')
plt.show()

