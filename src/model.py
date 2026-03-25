import torch
from torch import nn

from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


# Make device agnostic code. When we train on Colabs GPUs device will be cuda
device = "cuda" if torch.cuda.is_available() else "cpu" 
# print(device)

# Import the ResNet model, setup of loss function and the optimizers. Note that we use a pretrained model by using weights='DEFAULT'
# If we want to compare untrained, we simply leave it blank: model = models.resnet18()
model = models.resnet18(weights='DEFAULT').to(device)
#print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)
''' optimizer = torch.optim.Adam(params=model.parameters(),
                            lr=0.01) '''

# TODO: Setup of Muon optimizer. Not as simple as optimizer = torch.optim.Muon(params=model.parameters(), lr=0.01) as it need 2D-parameters

# TODO: The final layer is used for an output of 1000 classes due to the model's previous training. We need to figure out how to change that to 

# Training the model
epochs = 100

for epoch in range(epochs):
    model.train()

    # TODO: Figure out how to forward pass for ResNet

    # loss = loss_fn(x,y) # <-- Needs an input and target. Figure out what x, y can be
    # TODO: Write an accuracy funciton to see how the model improves on training data

    optimizer.zero_grad()
    # Once the loss can be calculated, we need backpropagation by using loss.backward()
    optimizer.step()

# Evaluating the model
with torch.no_grad():
    model.eval()

    # TODO: Forward pass

    # TODO: Calculate loss once we've figured out the paramameters to the loss function; test_loss = loss_fn(x, y)

    # TODO: Calculate accuracy; test_accuracy

# Uncomment this once loss and accuracy functions are implemented
'''if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")'''