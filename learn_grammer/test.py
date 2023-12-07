import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from torchvision.transforms import ToTensor

# print(torch.ones([])) # 每个tensor值为1
# print(torch.empty([])) # 返回一个未初始化的tensor
# print(torch.zeros([])) # 每个tensor值为0


# import torch
# import clip
# import numpy as np
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device) # shape(1,3,224,224)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device) # shape(3,77)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


# a = torch.tensor([1,2,3,4,5], dtype=torch.float)
# b = nn.init.normal_(a) # 样本方差
# print(b)
# len = a.shape[0]
# mean = torch.mean(a)
# var = torch.sqrt(torch.pow(a - mean, 2).sum() / (len - 1))
# c = (a - mean) / var
# d = a.std() # 样本标准差
# print(var)
# print(d)


# model = nn.LayerNorm(4) # 总体标准差
# x = torch.tensor(range(12), dtype=torch.float32).reshape(3, 4)
# print(model(x)) # 对最后一维进行标准化，shape(3,4)


# image = Image.open('CLIP.png')
# image = image.convert('RGB') # RGBA -> RGB
# image = ToTensor()(image)
# print(image.shape)


# from collections import OrderedDict
# d_model = 16
# x = OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ])
# print(x)
# for key in x.keys():
#     print(key)
#     print(x[key])


# a = torch.tensor([1,2,3,4], dtype=torch.float)
# print(a.norm(dim=0, keepdim=True)) # 总体方差


# loss_func = nn.CrossEntropyLoss()

# x = torch.rand(4, 5)
# y = torch.tensor([0,3,4,4])

# print(loss_func(x, y))


# from torch.nn.utils.rnn import pad_sequence
# data = [torch.rand(3, 12), torch.rand(4, 12), torch.rand(7, 12)]
# x = pad_sequence(data, batch_first=True)
# print(type(x))
# print(x.shape)
# print(x.dtype)
# print(torch.LongTensor([1, 2]).dtype)


# import numpy as np

# x = np.array([1, 2])
# y = np.array([1., 2.])
# print(x.dtype)
# print(y.dtype)

# b = ['a', 'fd', 's', 'd']
# a = torch.tensor([1,2,3])
# for i in range(len(a)):
#     print(b[a[i]])

# from torch.nn.functional import interpolate
# a = torch.rand(4, 3, 5)
# b = interpolate(a, 8)
# print(b.shape)
# c = a.norm(dim=-1)
# print(c.shape)
# c = a / c.unsqueeze(-1)
# print(torch.sum(c ** 2, dim=-1))

# a = torch.tensor([[1,4,2,1],
#                   [6,3,4,3],
#                   [4,3,2,9]])
# print(a.argmax(dim=-1))
# print(torch.cuda.is_available())

# print("hello world {}".format(round(3. / 7, 6)))


# from my_model import cross_entropy_loss

# model = nn.CrossEntropyLoss()
# my_model = cross_entropy_loss
# x = torch.arange(36).reshape((6, 6))
# x = x.type(torch.float32)
# y = torch.arange(len(x))

# print(model(x, y))
# print(my_model(x, y))
# print(model(x.t(), y))
# print(my_model(x.t(), y))


# import numpy as np
# import matplotlib.pyplot as plt

# Fs = 200
# f1 = 10
# f2 = 30
# x = np.arange(0.0, 1.0, 1/Fs)
# y = 2 * np.sin(2 * np.pi * f1 * x) + 5 * np.sin(2 * np.pi * f2 * x)

# plt.plot(x, y)
# plt.show()

# plt.specgram(y, Fs=Fs)
# plt.show()


# from collections import OrderedDict
# d = OrderedDict([('a', 1), ('c', 3), ('b', 2)])

# for key in list(d.keys()): # 先固定迭代对象
#     if key == 'a':
#         del d[key]
# print(d.keys())


x = torch.rand(4, 5)
print(x[:1].shape)
print(x[0,:].shape)
print(x.dim())