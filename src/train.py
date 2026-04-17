import model as m
import matplotlib.pyplot as plt
import time  # https://www.educative.io/answers/how-to-measure-elapsed-time-in-python
import seed


# opt_in_use = "adam"
# model, loss_fn, optimizer = get_model_opt_loss()
# model, loss_fn, optimizers = m.get_model_opt_loss(opt_name=opt_in_use, pretrain=False) <-- moving this down to the loop to reset for each optimizer
train_dataloader, test_dataloader, val_dataloader = m.get_dataloaders(batch_size=16)
optimizer_used = ["sgd", "adam", "muon"]
all_results = {}

for optim in optimizer_used:
    # seed.set_seed(1234567890)
    start_time = time.time()
    print(f"--- Training & Testing {optim} ---")
    model, loss_fn, optimizers = m.get_model_opt_loss(opt_name=optim, pretrain=False)

    optim_results = m.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizers=optimizers,
        # first_order_optimizer=first_order_optimizer,
        # second_order_optimizer=second_order_optimizer,
        epochs=15,
    )
    all_results[optim] = optim_results
    stop_time = time.time()
    execution_time = stop_time - start_time
    print(f"Execution Time for {optim}: {execution_time:.2f} seconds")

    _, test_acc = m.test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)
    
    print(f"Accuracy for test dataset {optim}: {test_acc:.2%}")


# Plot validation loss for all three optimizers
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for optim, results in all_results.items():
    plt.plot(
        results["val_loss"], label=f"{optim} Loss", marker="o"
    )  # https://huggingface.co/datasets/bird-of-paradise/muon-tutorial/blob/main/Muon.ipynb
# plt.plot(optim_results["val_loss"], color="r", label="val_loss")
# plt.plot(optim_results["val_acc"], color="g", label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title(f"Validation Loss vs Epoch for the SGD, Adam, and Muon Optimizers")
plt.grid(True)
plt.legend()
plt.show()

# Plot accuracy for all three optimizers: https://huggingface.co/datasets/bird-of-paradise/muon-tutorial/blob/main/Muon.ipynb & https://www.w3schools.com/python/matplotlib_subplot.asp
plt.subplot(1, 2, 2)
for optim, results in all_results.items():
    plt.plot(results["val_acc"], label=f"{optim} Accuracy", marker="o")
# plt.plot(optim_results["val_loss"], color="r", label="val_loss")
# plt.plot(optim_results["val_acc"], color="g", label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title(f"Validation Accuracy vs Epoch for the SGD, Adam, and Muon Optimizers")
plt.grid(True)
plt.legend()
plt.show()

