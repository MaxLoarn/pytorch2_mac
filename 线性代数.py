import random
import torch


def appendlist():
    listw = []
    for q in range(9):
        e = random.randint(0, 1)
        listw.append(e)
    qqqq = torch.tensor(listw).reshape(3, 3)
    print(qqqq)
    print('********************************')
    return listw


qwe = appendlist()
asd = appendlist()
zxc = appendlist()

qqq = [qwe, asd, zxc]

www = torch.tensor(qqq).reshape(3, 3, 3)
print('以下为由三个二维向量合并为一个三维向量进行转换：\n', www)


