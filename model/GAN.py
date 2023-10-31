import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.linear1 = nn.Sequential(
        nn.Linear(784, 1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
    )
    self.linear2 = nn.Sequential(
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
    )
    self.linear3 = nn.Sequential(
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
    )
    self.linear4 = nn.Sequential(
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    x = self.linear4(x)
    return x

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.linear1 = nn.Sequential(
        nn.Linear(128, 256),
        nn.LeakyReLU()
    )
    self.linear2 = nn.Sequential(
        nn.Linear(256, 512),
        nn.LeakyReLU()
    )
    self.linear3 = nn.Sequential(
        nn.Linear(512, 1024),
        nn.LeakyReLU()
    )
    self.linear4 = nn.Sequential(
        nn.Linear(1024, 784),
        nn.Tanh()
    )

  def forward(self, x):
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    x = self.linear4(x)
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)