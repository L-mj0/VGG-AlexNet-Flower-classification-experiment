import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


class AlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.features = models.alexnet(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def train(model, criterion, optimizer, num_epochs=25):
    for i in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in dataloader['train']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        print('epoch:%d,train loss:%.3f,train acc:%.3f%%' % (
        i + 1, running_loss / len(dataloader['train']), 100 * correct_train / total_train))

        epoch_loss = running_loss / len(dataloader['train'])
        epoch_acc = 100 * correct_train / total_train

        print(f'Epoch {i + 1}/{num_epochs} => '
              f'Training Loss: {epoch_loss:.4f} | '
              f'Training Accuracy: {epoch_acc:.4f}')


def test(model):
    model.eval()

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = 100 * correct_test / total_test
    print(f'Test Accuracy: {test_acc:.4f}')


# 定义数据预处理和加载器
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

data_dir = 'data'

image_datasets = {x: datasets.ImageFolder(root=data_dir + '/' + x, transform=data_transform[x]) for x in
                  ['train', 'test']}

dataloader = {x: DataLoader(dataset=image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in
              ['train', 'test']}



if __name__ == '__main__':

    # 定义模型
    num_classes = len(image_datasets['train'].classes)
    print(num_classes)
    model = AlexNet(num_classes=num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    train(model, criterion, optimizer, 50)

    test(model)
