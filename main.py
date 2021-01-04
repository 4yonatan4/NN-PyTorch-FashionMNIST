import sklearn as sklearn
import torch
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F

# Constants
IMAGE_SIZE = 28 * 28


class Model_A(nn.Module):
    def __init__(self, image_size):
        super(Model_A, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), -1)


def train_model(model, optimizer, criterion,
                nepochs, train_x, train_y, val_x, val_y):
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
    for e in range(nepochs):
        running_loss = 0
        running_val_loss = 0
        running_train_acc = 0
        running_val_acc = 0
        for image, label in zip(train_x, train_y):
            # Training pass
            model.train()  # set model in train mode
            optimizer.zero_grad()
            model_out = model(image)
            a = torch.reshape(label, (1,))
            loss = criterion(model_out, a.long())
            loss.backward()
            # one gradient descent step
            optimizer.step()
            running_loss += loss.item()
            y_hat = np.argmax(model_out.detach().numpy())
            if y_hat == label.item():
                running_train_acc += 1
        # Validation
        else:
            val_loss = 0
            # Evaluate model on validation at the end of each epoch.
            with torch.no_grad():
                for image, label in zip(val_x, val_y):
                    # Validation pass
                    model_out = model(image)
                    a = torch.reshape(label, (1,))
                    val_loss = criterion(model_out, a.long())
                    y_hat = np.argmax(model_out.detach().numpy())
                    if y_hat == label.item():
                        running_val_acc += 1
                    running_val_loss += val_loss

            # Track train loss and validation loss
            train_losses.append(running_loss / train_length)
            val_losses.append(running_val_loss / val_length)
            # Track train acc and validation acc
            train_acc.append(running_train_acc / train_length)
            val_acc.append(running_val_acc / val_length)
            print("Epoch: {}/{}.. ".format(e + 1, nepochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / train_length),
                  "Validation Loss: {:.3f}.. ".format(running_val_loss / val_length))
            print("Epoch: {}/{}.. ".format(e + 1, nepochs),
                  "Training Acc: {:.3f}.. ".format(running_train_acc / train_length),
                  "Validation Acc: {:.3f}.. ".format(running_val_acc / val_length))
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


def main():
    # Initial dataset and transforms
    # my_transforms, train_loader, test_loader = init()

    # Get data input and create numpy array
    # Test_x is just for the submit, we don't use it during debugging
    train_data_x, train_data_y, test_x = np.loadtxt(sys.argv[1]), np.loadtxt(sys.argv[2]), np.loadtxt(sys.argv[3])

    # Split the train data to: 80% train, 20% validation
    train_x, val_x, train_y, val_y = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=42)
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    val_x = val_x.astype(np.float32)
    val_y = val_y.astype(np.float32)

    # Create tensors from the data.
    train_x, val_x, train_y, val_y = torch.from_numpy(train_x), torch.from_numpy(val_x), torch.from_numpy(
        train_y), torch.from_numpy(val_y)

    # Choose loss function
    criterion = nn.NLLLoss()
    # ================================================== MODEL A ==================================================
    # Build the architecture of the network
    model_a = Model_A(image_size=IMAGE_SIZE)
    # Define the optimizer function and the value of the learning rate
    lr = 0.00001
    # Try to change to Adam, Adadelta, RMSprop
    optimizer = optim.SGD(model_a.parameters(), lr=lr)
    nepochs = 10
    # Train
    train_losses, val_losses, train_acc, val_acc = train_model(model_a, optimizer, criterion, nepochs, train_x, train_y,
                                                               val_x, val_y)

    # plot train and validation loss as a function of #epochs
    epochs = [*range(1, nepochs + 1)]
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, train_acc, label='Training Loss')
    plt.plot(epochs, val_acc, label='Validation Loss')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Test the model
    # test(model_a, test_x, test_y)


if __name__ == '__main__':
    main()
