import sklearn as sklearn
import torch
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from scipy import stats

# Constants
IMAGE_SIZE = 28 * 28
BATCH_SIZE = 200
NEPOCHS = 10
PERCENT = 0.8


class Model_A(nn.Module):
    '''
    Model A - Neural Network with two hidden layers.
    first layer - size 100 with ReLU Activation.
    second layer - size 50 with ReLU Activation.
    '''

    def __init__(self, ):
        super(Model_A, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), -1)


class Model_B(nn.Module):
    '''
        Model B - Neural Network with two hidden layers.
        first layer - size 100 with ReLU Activation.
        second layer - size 50 with ReLU Activation.
        '''

    def __init__(self):
        super(Model_B, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), -1)


class Model_C(nn.Module):
    '''
        Model C - Neural Network Dropout on Model A.
        first layer - size 100 with ReLU Activation then Dropout.
        second layer - size 50 with ReLU Activation then Dropout.
        '''

    def __init__(self):
        super(Model_C, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fd1 = torch.nn.Dropout(p=0.001)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = self.fd1(x)
        x = F.relu(self.fc0(x))
        x = self.fd1(x)
        x = F.relu(self.fc1(x))
        x = self.fd1(x)
        x = F.relu(self.fc2(x))
        x = self.fd1(x)
        return F.log_softmax(x, -1)


class Model_D(nn.Module):
    '''
        Model D - Adding Batch Normalization to Model A.
        first layer - size 100 with Batch Norm.
        second layer - size 50 with Batch Norm.
        '''

    def __init__(self):
        super(Model_D, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = torch.nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.batch_x1 = torch.nn.BatchNorm1d(100)
        self.batch_x2 = torch.nn.BatchNorm1d(50)
        self.batch_x3 = torch.nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = self.fc0(x)
        x = F.relu(self.batch_x1(x))
        x = self.fc1(x)
        x = F.relu(self.batch_x2(x))
        x = self.fc2(x)
        x = F.relu(self.batch_x3(x))
        return F.log_softmax(x, -1)


class Model_E(nn.Module):
    '''
        Model E - Neural Network with five hidden layers.
        first layer - size 128 with ReLU Activation.
        second layer - size 64 with ReLU Activation.
        third layer - size 10 with ReLU Activation.
        fourth layer - size 10 with ReLU Activation.
        fifth layer - size 10 with ReLU Activation.
        '''

    def __init__(self):
        super(Model_E, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = torch.nn.Linear(IMAGE_SIZE, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc0(x))  # Hidden layer 1
        x = F.relu(self.fc1(x))  # Hidden layer 2
        x = F.relu(self.fc2(x))  # Hidden layer 3
        x = F.relu(self.fc3(x))  # Hidden layer 4
        x = F.relu(self.fc4(x))  # Hidden layer 5
        return F.log_softmax(x, -1)


def shuffle(first_set, second_set):
    assert len(first_set) == len(second_set)
    permutation = np.random.permutation(len(first_set))
    return first_set[permutation], second_set[permutation]


def train_model(model, optimizer, criterion, train_x, train_y, val_x, val_y, name_of_model, batch_size):
    '''
    Train a pytorch model and evaluate it every epoch.
    Params:
    model - a pytorch model to train
    optimizer - an optimizer
    criterion - the criterion (loss function)
    nepochs - number of training epochs
    train_x - all images from the trainset
    train_y - all labels from the trainset
    val_x - all images from the validation set
    val_y - all labels from the validation set
    '''
    train_losses, val_losses, train_acc, val_acc = [], [], [], []
    train_length = len(train_x)
    val_length = len(val_x)
    print(f"Running now MODEL {name_of_model}:")

    for e in range(NEPOCHS):
        running_loss = 0
        running_val_loss = 0
        running_train_acc = 0
        running_val_acc = 0
        # training_set, lables_set = shuffle(train_x.dataset, train_y.dataset)
        for batch_idx, (image, label) in enumerate(zip(train_x, train_y)):
            # Training pass
            model.train()  # set model in train mode
            optimizer.zero_grad()
            model_out = model(image)
            a = torch.reshape(label, (batch_size,))
            loss = criterion(model_out, a.long())
            # loss = F.nll_loss(model_out, a.long())
            loss.backward()
            # one gradient descent step
            optimizer.step()
            running_loss += loss.item()
            temp = model_out.detach().numpy()
            # label = label.detach().numpy()
            for i in range(0, len(temp)):
                y_hat = np.argmax(temp[i])
                if y_hat == label[i]:
                    running_train_acc += 1
        # Validation
        else:
            val_loss = 0
            # Evaluate model on validation at the end of each epoch.
            with torch.no_grad():
                for image, label in zip(val_x, val_y):
                    # Validation pass
                    model_out = model(image)
                    a = torch.reshape(label, (batch_size,))
                    val_loss = criterion(model_out, a.long())
                    temp = model_out.detach().numpy()
                    # label = label.detach().numpy()
                    for i in range(0, len(temp)):
                        y_hat = np.argmax(temp[i])
                        if y_hat == label[i]:
                            running_val_acc += 1
                    running_val_loss += val_loss
            # Track train loss and validation loss
            train_losses.append(running_loss / (train_length * batch_size))
            val_losses.append(running_val_loss / (val_length * batch_size))
            # Track train acc and validation acc
            train_acc.append(running_train_acc / (train_length * batch_size))
            val_acc.append(running_val_acc / (val_length * batch_size))
            print("Epoch: {}/{}.. ".format(e + 1, NEPOCHS),
                  "Training Loss: {:.3f}.. ".format(running_loss / (train_length * batch_size)),
                  "Validation Loss: {:.3f}.. ".format(running_val_loss / (val_length * batch_size)))
            print("Epoch: {}/{}.. ".format(e + 1, NEPOCHS),
                  "Training Acc: {:.3f}.. ".format(running_train_acc / (train_length * batch_size)),
                  "Validation Acc: {:.3f}.. ".format(running_val_acc / (val_length * batch_size)))
    return train_losses, val_losses, train_acc, val_acc


def test(model, test_x, test_y):
    # We say to the model that we are in test situation.
    model.eval()
    test_loss = 0
    correct = 0
    # In test situation we don't need all the calculates of the derivatives and more,
    # So here we say to the model to not do all this calculates
    with torch.no_grad():
        for data, target in zip(test_x, test_y):
            # Forward the x in the network
            output = model(data)
            y_hat = np.argmax(output)
            if y_hat.item() == target.item():
                correct += 1
            # Reduce give us one value instead of 10
            # test_loss += F.nll_loss(output, target, size_average=False, reduce=True).item()
            # Function max return us - (max, arg_max), we want the arg max
            # pred = output.max(1, keepdim=True)[1]
            # correct += pred.eq(target.view_as(pred)).cpu().sum()
    data_length = len(test_x)
    print(correct / data_length)
    # test_loss /= data_length
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    # test_loss, correct, data_length,
    # 100. * correct / data_length


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

def plot(train_losses, val_losses, train_acc, val_acc, nepochs):
    # plot train and validation loss as a function of #epochs
    epochs = [*range(1, nepochs + 1)]
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def split_data(x_data, percent):
    test_size = int(percent * x_data.shape[0])
    train_data = x_data[:test_size]
    test_data = x_data[test_size:]
    return train_data, test_data


def main():
    # Initial dataset and transforms
    # my_transforms, train_loader, test_loader = init()

    # Get data input and create numpy array
    # Test_x is just for the submit, we don't use it during debugging
    train_data_x, train_data_y, test_x = np.loadtxt(sys.argv[1]), np.loadtxt(sys.argv[2]), np.loadtxt(sys.argv[3])
    train_data_x = stats.zscore(train_data_x)
    new_data = np.column_stack((train_data_x, train_data_y))
    np.random.shuffle(new_data)
    temptrain_x, tempval_x = split_data(new_data, PERCENT)
    val_x = tempval_x[:, :IMAGE_SIZE]
    val_y = tempval_x[:, IMAGE_SIZE]
    train_x = temptrain_x[:, :IMAGE_SIZE]
    train_y = temptrain_x[:, IMAGE_SIZE]

    # Split the train data to: 80% train, 20% validation
    # train_x, val_x, train_y, val_y = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=42)

    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_x = val_x.astype(np.float32)
    val_y = val_y.astype(np.float32)
    # Create tensors from the data
    # train_x, val_x, train_y, val_y = torch.from_numpy(train_x), torch.from_numpy(val_x), torch.from_numpy(
    #    train_y), torch.from_numpy(val_y)
    train_x = DataLoader(dataset=train_x, batch_size=BATCH_SIZE)
    val_x = DataLoader(dataset=val_x, batch_size=BATCH_SIZE)
    train_y = DataLoader(dataset=train_y, batch_size=BATCH_SIZE)
    val_y = DataLoader(dataset=val_y, batch_size=BATCH_SIZE)
    # Choose loss function
    criterion = nn.NLLLoss()
    # ================================================== MODEL A ==================================================
    # Build the architecture of the network
    model_a = Model_A()
    # Define the optimizer function and the value of the learning rate
    lr = 0.1
    # SGD optimizer
    optimizer = optim.SGD(model_a.parameters(), lr=lr)
    # Train
    train_losses, val_losses, train_acc, val_acc = train_model(model_a, optimizer, criterion, train_x, train_y,
                                                                val_x, val_y, name_of_model='A', batch_size=BATCH_SIZE)
    # plot train and validation loss as a function of #epochs
    plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS)

    # ================================================== MODEL B ==================================================
    # Build the architecture of the network
    model_b = Model_B()
    # Define the optimizer function and the value of the learning rate
    lr = 0.001
    # SGD optimizer
    optimizer = optim.Adam(model_b.parameters(), lr=lr)
    # Train
    train_losses, val_losses, train_acc, val_acc = train_model(model_b, optimizer, criterion, train_x, train_y,
                                                                val_x, val_y, name_of_model='B', batch_size=BATCH_SIZE)
    # plot train and validation loss as a function of #epochs
    plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS)

    # ================================================== MODEL C ==================================================
    # Build the architecture of the network
    model_c = Model_C()
    # Define the optimizer function and the value of the learning rate
    lr = 0.001
    # SGD optimizer
    optimizer = optim.Adagrad(model_c.parameters(), lr=lr)
    # Train
    train_losses, val_losses, train_acc, val_acc = train_model(model_c, optimizer, criterion, train_x, train_y,
                                                                val_x, val_y, name_of_model='C', batch_size=BATCH_SIZE)
    # plot train and validation loss as a function of #epochs
    plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS)

    # ================================================== MODEL D ==================================================
    # Build the architecture of the network
    model_d = Model_D()
    # Define the optimizer function and the value of the learning rate
    lr = 0.01
    # SGD optimizer
    optimizer = optim.Adagrad(model_d.parameters(), lr=lr)
    # Train
    train_losses, val_losses, train_acc, val_acc = train_model(model_d, optimizer, criterion, train_x, train_y,
                                                                val_x, val_y, name_of_model='D', batch_size=BATCH_SIZE)
    # plot train and validation loss as a function of #epochs
    plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS)

    # ================================================== MODEL E ==================================================
    # Build the architecture of the network
    model_e = Model_E()
    # Define the optimizer function and the value of the learning rate
    lr = 0.001
    # SGD optimizer
    optimizer = optim.Adam(model_e.parameters(), lr=lr)
    # Train
    train_losses, val_losses, train_acc, val_acc = train_model(model_e, optimizer, criterion, train_x, train_y,
                                                                val_x, val_y, name_of_model='E', batch_size=BATCH_SIZE)
    # plot train and validation loss as a function of #epochs
    plot(train_losses, val_losses, train_acc, val_acc, NEPOCHS)


if __name__ == '__main__':
    main()
