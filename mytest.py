import torch

a = torch.arange(13)   # 生成[0,1,2,...,11,12]
a = a.repeat(13, 1)   # 生成13*13矩阵，每一行都是[0,1,2,...,11,12]
# a = a.t()   # 转置
a = a.view([1, 1, 13, 13])   # 生成1*1*13*13


b = torch.rand(64, 3, 13, 13)
b = b.fill_(1)

c = torch.rand(64, 4)
c = c*13

d = '/JPEGImages/'
d = d.replace('JPEGImages', 'labels')
print(d)