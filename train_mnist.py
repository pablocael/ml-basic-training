from torch import nn

from torch.optim import Adam
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import numpy as np

class MySimpleModel(nn.Module):
    def __init__(self, input_size):

        super(MySimpleModel, self).__init__()
        self._layers = nn.Sequential(
            nn.Linear(input_size, input_size//2, bias=True),
            nn.ReLU(),
            nn.Linear(input_size//2,  input_size//4, bias=True),
            nn.ReLU(),
            nn.Linear(input_size//4,  input_size//8, bias=True),
            nn.ReLU(),
            nn.Linear(input_size//8,  input_size//16, bias=True),
            nn.ReLU(),
            nn.Linear(input_size//16,  10, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        return self._layers(X)


if __name__ == '__main__':

    mnist_dataset = datasets.MNIST(root='./data/', download=True, transform=ToTensor())

    smp, lbl = mnist_dataset[0]
    print(smp.shape, lbl)


    N = len(mnist_dataset)

    train_proportion = 0.8

    train_N = int(train_proportion * N)
    test_N = N - train_N

    mnist_train_dataset, mnist_test_dataset = random_split(mnist_dataset, [train_N, test_N])

    num_epochs = 100
    batch_size = 16

    mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=True)

    model = MySimpleModel(28*28)

    criterion = nn.NLLLoss()

    optimizer = Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):

        model.train()
        # 1 run an epoch for the training data
        for batch, labels in mnist_train_dataloader:

            flatten_batch = batch.view(batch.shape[0], -1)
            optimizer.zero_grad()

            prediction = model(flatten_batch)

            loss = criterion(prediction, labels)
            loss.backward()

            optimizer.step()

        # 2 eval the model using the test dataset to check our current model accuracy
        model.eval()

        overall_loss = np.inf
        accuracy = 0
        num_examples = 0
        for batch, labels in mnist_test_dataloader:

            flatten_batch = batch.view(batch.shape[0], -1)
            prediction = model(flatten_batch)

            loss = criterion(prediction, labels)
            preds = np.argmax(prediction.detach().numpy(), axis=1)
            local_acc = np.sum(preds == np.int64(labels)) / prediction.shape[0]
            accuracy += local_acc

            overall_loss += loss.item()
            num_examples += 1

        overall_loss /= test_N
        avg_accuracy = accuracy / num_examples

        print(f'> Current accuracy for epoch {epoch} is {avg_accuracy}')







