import torch


# Neural Network
class Cnn(torch.nn.Module):
    def __init__(self, name, d_in_sqrt, d_out, dtype=torch.float, device='cpu'):
        super(Cnn, self).__init__()

        self.dtype = dtype
        self.device = device

        self.name = name

        self.activate = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(d_out)

        # Input channels = 1, output channels = 6
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(6 * (d_in_sqrt/2) * (d_in_sqrt/2), 32)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(32, 10)

    def forward(self, z):
        # Computes the activation of the first convolution
        # Size changes from (1, 16, 16) to (6, 16, 16)
        z = self.activate(self.conv1(z))

        # Size changes from (6, 16, 16) to (6, 8, 8)
        z = self.pool(z)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (6, 8, 8) to (1, 384)
        # Recall that the -1 infers this dimension from the other given dimension
        z = z.view(-1, 6 * 8 * 8)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 384) to (1, 32)
        z = self.activate(self.fc1(z))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 32) to (1, 10)
        z = self.fc2(z)
        return (z)


    def predict_prob(self, x):
        z = self.forward(x)
        prob = self.softmax(z)
        return (prob)

    def predict(self, x):
        z = self.forward(x)
        pred = torch.max(z.data, 1)[1]
        return (pred)

