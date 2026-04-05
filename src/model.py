import torch
from torch import nn

from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from pathlib import Path

# from muon import (
#     SingleDeviceMuonWithAuxAdam,
# ) # https://github.com/KellerJordan/Muon/tree/master

from muon_modified import Muon

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42) # Attempt to get the same result each run. From section 3.4 https://www.learnpytorch.io/06_pytorch_transfer_learning/

# Make device agnostic code. When we train on Colabs GPUs device will be cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # It can be removed when benchmarking on the cloud
else:
    device = torch.device("cpu")
print(device)


def get_dataloaders(batch_size: int = 16):
    # See https://www.learnpytorch.io/04_pytorch_custom_datasets/ which served as a guide for this process
    project_root = Path(__file__).resolve().parent.parent
    # From section 2:
    # train_dir = project_root / "processed_dataset" / "train"
    # test_dir = project_root / "processed_dataset" / "test"
    # val_dir = project_root / "processed_dataset" / "val"

    train_dir = project_root / "original_dataset" / "train"
    test_dir = project_root / "original_dataset" / "test"
    val_dir = project_root / "original_dataset" / "val"

    # Convert to tensors, from section 3.1
    data_transform = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=3
            ),  # The data we have sorted out is a single-channel 8-bit grayscale png picture
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10)
        ]
    )

    # transforms.RandomHorizontalFlip(p=0.5) Random rotation may not necessarily be useful or even have negative effects on the brain tumor images we use

    # From section 4.
    train_data = datasets.ImageFolder(
        root=train_dir, transform=data_transform, target_transform=None
    )

    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

    val_data = datasets.ImageFolder(root=val_dir, transform=data_transform)

    # class_names = train_data.classes
    # print(class_names)
    # class_dict = train_data.class_to_idx
    # print(class_dict)
    # print(len(train_data), len(test_data))

    # From section 4.1
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False
    )

    val_dataloader = DataLoader(
        dataset=val_data, batch_size=batch_size, shuffle=False
    )  # https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets

    return train_dataloader, test_dataloader, val_dataloader


# TODO: Setup of Muon optimizer. Not as simple as optimizer = torch.optim.Muon(params=model.parameters(), lr=0.01) as it need 2D-parameters
# Tried with this function to separate the parameters, but decided to simply move it into the get_model_opt_loss-function
""" def separate_params():
    # This needs work. Practically pseudo code right now
    muon_params = []
    first_order_params = []
    # Separating >= 2D-tensors that will be used by Muon, and 1D-tensors used by Adam/SGD
    for param in model.parameters():
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            first_order_params.append(param)
    
    return muon_params, first_order_params"""


def get_model_opt_loss():  # May allow selecting optimizer via string   
    # Import the ResNet model, setup of loss function and the optimizers. Note that we use a pretrained model by using weights='DEFAULT'
    # If we want to compare untrained, we simply leave it blank: model = models.resnet18()
    # model = models.resnet18(weights="DEFAULT")
    model = models.resnet18(weights=None)
    # print(model)

    # The model has a final layer, fc, which has out_features=1000: (fc): Linear(in_features=512, out_features=1000, bias=True). The dataset we use only has
    # three categories --> three potential outputs in the final layer - cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
    # The way to change the final fc-layer was found on https://discuss.pytorch.org/t/resnet-last-layer-modification/33530
    # Notice that they use a sequential layer. I've kept ours linear since that was the original ResNet18 architecture. Should we have overfitting issues, adding dropout could be useful
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 3)) # changed from nn.Linear(num_ftrs, 3)
    model = model.to(device)

    # print(model)

    muon_params = []
    first_order_params = []

    # Separating >= 2D-tensors that will be used by Muon, and 1D-tensors used by Adam/SGD. TODO: Double check the logic for param.ndim == 4. Should perhaps be appended to muon_params?
    # for param in model.parameters():
    #     if param.ndim == 2:
    #         muon_params.append(param)
    #     elif param.ndim == 4:
    #         # param.view(param.shape[0], -1)  # Flatten all but first dim, solution from https://huggingface.co/datasets/bird-of-paradise/muon-tutorial/blob/main/Muon.ipynb. Possibly not wise to have this in the for-loop
    #         muon_params.append(param)
    #     else:
    #         first_order_params.append(param)

    for name, param in model.named_parameters():
        if "fc" in name or "conv1" in name:  # Jordan's advice
            first_order_params.append(param)
            continue
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            first_order_params.append(param)

    # Muon implementation by keller Jordan https://github.com/KellerJordan/Muon/tree/master
    """ param_groups = [
        dict(params=muon_params, use_muon=True, lr=0.02, weight_decay=0.01),
        dict(
            params=first_order_params,
            use_muon=False,
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        ),
    ] """

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0003)
    # second_order_optimizer = torch.optim.Muon(params=muon_params, lr=0.01)
    second_order_optimizer = Muon(
        params=muon_params,
        lr=0.001,
        # weight_decay=0.2,
        # momentum=0.7,
    )
    first_order_optimizer = torch.optim.Adam(params=first_order_params, lr=0.0003)

    # optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    return model, loss_fn, first_order_optimizer, second_order_optimizer  # , optimizer


# Training the model
# ------- From section 7.5 -----------
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    # optimizer: torch.optim.Optimizer,
    first_order_optimizer: torch.optim.Optimizer,
    second_order_optimizer: torch.optim.Optimizer,
):
    size = len(dataloader.dataset)

    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # 3. Optimizer zero grad
        # optimizer.zero_grad()
        first_order_optimizer.zero_grad()
        second_order_optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        # optimizer.step()
        first_order_optimizer.step()
        second_order_optimizer.step()

        # Batch level
        # if batch % 10 == 0:
        #     batch_loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {batch_loss: > 7f} [{current:>5d}/{size:>5d}]")

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
):
    # Put model in eval mode
    model.eval()
    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_y_pred = model(X)
            # 2. Calculate and accumulate loss
            loss = loss_fn(test_y_pred, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_y_pred.argmax(dim=1)
            val_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    # test_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    # optimizer: torch.optim.Optimizer,
    first_order_optimizer=torch.optim.Optimizer,
    second_order_optimizer=torch.optim.Optimizer,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 30,
):

    # 2. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            # optimizer=optimizer,
            first_order_optimizer=first_order_optimizer,
            second_order_optimizer=second_order_optimizer,
        )
        val_loss, val_acc = test_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn
        )

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )
    return results


# ------------------------------------

""" ——————————————————— Test ——————————————————— """
train_dataloader, test_dataloader, val_dataloader = get_dataloaders()
# model, loss_fn, optimizer = get_model_opt_loss()
model, loss_fn, first_order_optimizer, second_order_optimizer = get_model_opt_loss()

# train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
# print("loss:", train_loss)
# print("acc:", train_acc)

train(  # should this still be called train or do we name it def test?
    model=model,
    train_dataloader=train_dataloader,  # Since this is in the test-part, should we have test_dataloader?
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    # optimizer=optimizer,
    first_order_optimizer=first_order_optimizer,
    second_order_optimizer=second_order_optimizer,
    epochs=15,
)

""" ———————————————————————————————————————————— """
# epochs = 100

# for epoch in range(epochs):
#     model.train()

#     # TODO: Figure out how to forward pass for ResNet

#     # loss = loss_fn(x,y) # <-- Needs an input and target, such as a X_test, y_test etc
#     # TODO: Write an accuracy funciton to see how the model improves on training data

#     optimizer.zero_grad()
#     # Once the loss can be calculated, we need backpropagation by using loss.backward()
#     optimizer.step()

#     # Evaluating the model
#     with torch.no_grad():
#         model.eval()

#         # TODO: Forward pass

#         # TODO: Calculate loss once we've figured out the paramameters to the loss function; val_loss = loss_fn(x, y)

#         # TODO: Calculate accuracy; val_accuracy

#     # Uncomment this once loss and accuracy functions are implemented
#     """if epoch % 10 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {val_loss:.5f}, Test acc: {val_acc:.2f}%")"""
