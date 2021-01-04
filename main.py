import torch
import sys
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

# Constants
IMAGE_SIZE = 28 * 28

class MyNet(nn.Module):
    def __init__(self, image_size):
        super(MyNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


def train(epoch, model, optimizer, train_x, train_y):
    # We say to the model that we are in train situation.
    model.train()
    # for batch_idx, data, labels in zip(train_x, train_y):
    for data, labels in zip(train_x, train_y):
        # every weight has a value and grad value.
        # So we delete the last grad value from the last iterate
        optimizer.zero_grad()
        # the size of the output is: (batch_size, 10)
        output = model(data)
        # Get the loss
        loss = F.nll_loss(output, labels)
        # Back propagation.
        # Every parameter get his gradient (Calculate all the derivatives)
        loss.backward()
        # Update the weight with the new derivatives
        # Optimizer don't care how you calculate the derivatives
        optimizer.step()


def test(model, test_loader):
    # We say to the model that we are in test situation.
    model.eval()
    test_loss = 0
    correct = 0
    # In test situation we don't need all the calculates of the derivatives and more,
    # So here we say to the model to not do all this calculates
    with torch.no_grad():
        for data, target in test_loader:
            # Forward the x in the network
            output = model(data)
            # Reduce give us one value instead of 10
            test_loss += F.nll_loss(output, target, size_average=False, reduce=True).item()
            # Function max return us - (max, arg_max), we want the arg max
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


# def init():
#     my_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.30831,))
#     ])
#     train_loader = torch.utils.data.DataLoader(
#         datasets.FashionMNIST('./data', train=True, download=True, transform=my_transforms),
#         batch_size=64, shuffle=True
#     )
#     test_loader = torch.utils.data.DataLoader(
#         datasets.FashionMNIST('./data', train=False, transform=my_transforms),
#         batch_size=64, shuffle=False
#     )
#     return my_transforms, train_loader, test_loader


def main():
    # Initial dataset and transforms
    # my_transforms, train_loader, test_loader = init()

    # get the names of the arguments
    train_x, train_y, test_x = np.loadtxt(sys.argv[1]), np.loadtxt(sys.argv[2]), np.loadtxt(sys.argv[3])
    train_x, train_y, test_x = torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(test_x)

    # Split ti 80:20
    torch.split(train_x)
    # sklearn.model_selection.train_test_split()

    # Build the architecture of the network
    model = MyNet(image_size=IMAGE_SIZE)
    # Define the optimizer function and the value of the learning rate
    lr = 0.01
    # Try to change to Adam, Adadelta, RMSprop
    optimizer = optim.SGD(model.parameters(), lr=lr)
    epoch = 10
    # Train
    train(epoch, model, optimizer, train_x, train_y)
    # Test the model
    test(model, test_x)


if __name__ == '__main__':
    main()
