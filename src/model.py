import torch
from torch import nn

from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from pathlib import Path

# Make device agnostic code. When we train on Colabs GPUs device will be cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # It can be removed when benchmarking on the cloud
else:
    device = torch.device("cpu")
# print(device)


def get_dataloaders(batch_size: int = 64):
    # See https://www.learnpytorch.io/04_pytorch_custom_datasets/ which served as a guide for this process
    project_root = Path(__file__).resolve().parent.parent
    # From section 2:
    # train_dir = r"C:\Users\mikae\Documents\Exjobb\IA150X\processed_dataset\train" #TODO: Find out if there's a smart way to make this more dynamic
    train_dir = project_root / "processed_dataset" / "train"
    # test_dir = r"C:\Users\mikae\Documents\Exjobb\IA150X\processed_dataset\test"
    test_dir = project_root / "processed_dataset" / "test"

    val_dir = project_root / "processed_dataset" / "val"
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


def get_model_opt_loss():  # May allow selecting optimizer via string
    # Import the ResNet model, setup of loss function and the optimizers. Note that we use a pretrained model by using weights='DEFAULT'
    # If we want to compare untrained, we simply leave it blank: model = models.resnet18()
    model = models.resnet18(weights="DEFAULT")
    # print(model)

    # The model has a final layer, fc, which has out_features=1000: (fc): Linear(in_features=512, out_features=1000, bias=True). The dataset we use only has
    # three categories --> three potential outputs in the final layer - cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
    # The way to change the final fc-layer was found on https://discuss.pytorch.org/t/resnet-last-layer-modification/33530
    # Notice that they use a sequential layer. I've kept ours linear since that was the original ResNet18 architecture. Should we have overfitting issues, adding dropout could be useful
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    # print(model)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    """ optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01) """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    return model, optimizer, loss_fn


# TODO: Setup of Muon optimizer. Not as simple as optimizer = torch.optim.Muon(params=model.parameters(), lr=0.01) as it need 2D-parameters


# Training the model
# From section 7.5
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()

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
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 5,
):

    # 2. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
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

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        # results["train_loss"].append(
        #     train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        # )
        # results["train_acc"].append(
        #     train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
        # )
        # results["val_loss"].append(
        #     val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        # )
        # results["val_acc"].append(
        #     val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc
        # )

    # 6. Return the filled results at the end of the epochs
    return results


""" ——————————————————— Test ——————————————————— """
train_dataloader, test_dataloader, val_dataloader = get_dataloaders()
model, optimizer, loss_fn = get_model_opt_loss()

# train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
# print("loss:", train_loss)
# print("acc:", train_acc)

train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=10,
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
