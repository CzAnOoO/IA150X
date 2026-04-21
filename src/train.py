import model as m
import matplotlib.pyplot as plt
import time  # https://www.educative.io/answers/how-to-measure-elapsed-time-in-python
import seed


# opt_in_use = "adam"
# model, loss_fn, optimizer = get_model_opt_loss()
# model, loss_fn, optimizers = m.get_model_opt_loss(opt_name=opt_in_use, pretrain=False) <-- moving this down to the loop to reset for each optimizer
train_dataloader, test_dataloader, val_dataloader = m.get_dataloaders(batch_size=32) #adjusting 16, 32, 64, 128, 256
optimizer_used = ["sgd", "adam", "muon"]
all_results = {}

num_runs = 5
num_epochs = 15
run = 1

for optim in optimizer_used:
    print(f"------------------------ Run: {run} ------------------------")
    run += 1
    total_results = {"val_loss": [0] * num_epochs, "val_acc": [0] * num_epochs}
    total_test_acc = 0
    total_exec_time = 0

    for i in range(num_runs):
        seed.set_seed(1234567890 + i)
        print(f"--- Training & Testing {optim} ---")
        start_time = time.time()
        model, loss_fn, optimizers = m.get_model_opt_loss(
            opt_name=optim, pretrain=False
        )
        optim_results = m.train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            optimizers=optimizers,
            # first_order_optimizer=first_order_optimizer,
            # second_order_optimizer=second_order_optimizer,
            epochs=num_epochs,
        )
        stop_time = time.time()
        # all_results[optim] = optim_results
        execution_time = stop_time - start_time

        _, test_acc = m.test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn
        )
        total_test_acc += test_acc
        total_exec_time += execution_time

        for j in range(num_epochs):
            # https://realpython.com/python-dicts/
            total_results["val_loss"][j] += optim_results["val_loss"][j]
            total_results["val_acc"][j] += optim_results["val_acc"][j]

    all_results[optim] = {
        # Leverage List Comprehensions: https://realpython.com/list-comprehension-python/
        "val_loss": [x / num_runs for x in total_results["val_loss"]],
        "val_acc": [x / num_runs for x in total_results["val_acc"]],
    }

    print(f"Average execution time for {optim}: {total_exec_time / num_runs:.2f} s")
    print(f"Average accuracy for test dataset {optim}: {total_test_acc / num_runs:.2%}")

x = list(range(1, num_epochs + 1))

# Plot validation loss for all three optimizers
# plt.figure(figsize=(9, 3))
plt.subplot(2, 1, 1)
for optim, results in all_results.items():
    # https://huggingface.co/datasets/bird-of-paradise/muon-tutorial/blob/main/Muon.ipynb
    plt.plot(x, results["val_loss"], label=f"{optim} Loss", marker="o")
# plt.plot(optim_results["val_loss"], color="r", label="val_loss")
# plt.plot(optim_results["val_acc"], color="g", label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title(f"Validation Loss vs Epoch")  # for the SGD, Adam, and Muon Optimizers")
plt.xticks(x)
plt.grid(True)
plt.legend()
# plt.show()

# Plot accuracy for all three optimizers: https://huggingface.co/datasets/bird-of-paradise/muon-tutorial/blob/main/Muon.ipynb & https://www.w3schools.com/python/matplotlib_subplot.asp
plt.subplot(2, 1, 2)
for optim, results in all_results.items():
    plt.plot(x, results["val_acc"], label=f"{optim} Accuracy", marker="o")
# plt.plot(optim_results["val_loss"], color="r", label="val_loss")
# plt.plot(optim_results["val_acc"], color="g", label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title(f"Validation Accuracy vs Epoch")  # for the SGD, Adam, and Muon Optimizers")
plt.xticks(x)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("results.pdf")
plt.show()

