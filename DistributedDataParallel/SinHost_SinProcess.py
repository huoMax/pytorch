import time
import torch
import torchvision
import torch.utils.data.distributed
from torchvision import transforms


def main():
    # 定义transforms
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    # 选择 Pytorch 自带的 MNIST 分类数据集
    # 加载训练集
    data_set_train = torchvision.datasets.MNIST("../data", train=True, transform=trans, target_transform=None, download=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set_train, batch_size=256)
    print("训练集大小: {}".format(len(data_set_train)))

    # 加载测试集
    data_set_test =  torchvision.datasets.MNIST("../data", train=False, transform=trans, target_transform=None, download=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_set_test, batch_size=256)
    print("测试集大小: {}".format(len(data_set_test)))

    # 使用 ResNet 模型
    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)

    # 定义 loss 函数
    criterion = torch.nn.CrossEntropyLoss()

    # 定义 optimizer
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    # 训练
    for epoch in range(10):
        for i, data in enumerate(data_loader_train, 0):
            # 获取数据
            images, labels = data

            # 清除梯度积累
            opt.zero_grad()

            # forward
            outputs = net(images)

            # backward
            loss = criterion(outputs, labels)
            loss.backward()

            # optimizer
            opt.step()

            if i % 10 == 0:
                print("loss: {}     time: {}    loop: {}".format(loss.item(), time.time(), i))

    # 保存checkpoint
    torch.save(net, "./ModelSave/SinHost_SinProcess.pth")


if __name__ == "__main__":
    main()