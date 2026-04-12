import model as m
import matplotlib.pyplot as plt

train_dataloader, test_dataloader, val_dataloader = m.get_dataloaders(batch_size=32)
opt_in_use = "adam"
# model, loss_fn, optimizer = get_model_opt_loss()
model, loss_fn, optimizers = m.get_model_opt_loss(opt_name=opt_in_use, pretrain=False)

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

plt.figure(figsize=(12, 5))
plt.plot(optim_results["val_loss"], color="r", label="val_loss")
plt.plot(optim_results["val_acc"], color="g", label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Val")
plt.title(f"Loss & Accuracy vs Epoch for the {opt_in_use.upper()} Optimizer")
plt.legend()
plt.show()
