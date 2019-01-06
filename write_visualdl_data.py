from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import datetime
import argparse

from visualdl import LogWriter

log_writer = LogWriter("./log", sync_cycle=1000)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, train_losses):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward pass.
        output = model(data)

        # Negative log-likelihood
        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data),
                                                                           # Number of training example seen so far
                                                                           len(train_loader.dataset),
                                                                           # Total number of training examples
                                                                           100. * batch_idx / len(train_loader),
                                                                           # Percentage of training examples
                                                                           loss.item()))
            train_losses.add_record(epoch, float(loss.item()))


def test(args, model, device, test_loader, epoch, test_losses, test_accuracies):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # torch.onnx.export(model, test_loader.dataset, "pytorch_mnist_{}.onnx".format(epoch))

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))

    test_losses.add_record(epoch, float(test_loss))
    test_accuracies.add_record(epoch, float(test_accuracy))


def to_time(seconds):
    return str(datetime.timedelta(seconds=seconds))


def get_mnist_dataset(train=True, download=False):
    # Apply two transforms to the data.
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])
    return datasets.MNIST('./data', train=train, download=download, transform=t)


def get_argument_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    return parser.parse_args()


def main():
    # Training settings
    args = get_argument_parser()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(get_mnist_dataset(download=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(get_mnist_dataset(train=False),
                                              batch_size=args.test_batch_size,
                                              shuffle=True, **kwargs)

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    t0 = time.time()

    with log_writer.mode("train") as logger:
        train_losses = logger.scalar("scalars/train_loss")

    with log_writer.mode("test") as logger:
        test_losses = logger.scalar("scalars/test_loss")
        test_accuracies = logger.scalar("scalars/test_accuracy")

    for epoch in range(1, args.epochs + 1):
        t1 = time.time()

        train(args, model, device, train_loader, optimizer, epoch, train_losses)
        test(args, model, device, test_loader, epoch, test_losses, test_accuracies)

        print('Epoch lasted: {} seconds'.format(to_time(time.time() - t1)))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    print('Total time: {} seconds'.format(to_time(time.time() - t0)))


if __name__ == '__main__':
    main()
