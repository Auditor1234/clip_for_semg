from matplotlib import pyplot as plt
from load_mat import *
from PIL import Image


filename = 'dataset/DB2_E3/S1_E3_A1.mat'
movements, labels = load_data_from_file(filename)

movement = np.array(movements[1]).T[0]
plt.plot(movement)
plt.title('Signal')
plt.ylabel('amplitude')
plt.xlabel('time')
plt.savefig('dataset/img/signal_1_0.png', bbox_inches = 'tight')
plt.show()
