import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
# train_sampler = torch.utils.data.DataLoader(dataset=train_set)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4,
                                           num_workers=0,pin_memory=True)

# test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4,
#                                            num_workers=2,pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10):
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # running_loss += loss.item()
        # if i % 2000 == 1999:
        #     print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
        #     running_loss = 0.0

    end_time = time.time()
    print("[start_time: %f,end_time: %f" % (start_time, end_time))

print("Finish training")

