import model as m
import matplotlib.pyplot as plt
import time # https://www.educative.io/answers/how-to-measure-elapsed-time-in-python

train_dataloader, test_dataloader, val_dataloader = m.get_dataloaders(batch_size=32)
#opt_in_use = "adam"
# model, loss_fn, optimizer = get_model_opt_loss()
# model, loss_fn, optimizers = m.get_model_opt_loss(opt_name=opt_in_use, pretrain=False) <-- moving this down to the loop to reset for each optimizer

optimizer_used = ["SGD", "Adam", "Muon"]
all_results = {}

for optim in optimizer_used:
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
        epochs=10,
    )
    all_results[optim] = optim_results
    stop_time = time.time()
    execution_time = stop_time - start_time
    print(f"Execution Time for {optim}: {execution_time:.2f} seconds") # https://www.geeksforgeeks.org/python/how-to-get-two-decimal-places-in-python/


plt.figure(figsize=(12, 5))
for optim, results in all_results.items():
    plt.plot(results["val_loss"], label=f"{optim} Loss", marker="o") # https://huggingface.co/datasets/bird-of-paradise/muon-tutorial/blob/main/Muon.ipynb 
#plt.plot(optim_results["val_loss"], color="r", label="val_loss")
#plt.plot(optim_results["val_acc"], color="g", label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title(f"Validation Loss vs Epoch for the SGD, Adam, and Muon Optimizers")
plt.grid(True) 
plt.legend()
plt.show()
