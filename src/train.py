import model as m

train_dataloader, test_dataloader, val_dataloader = m.get_dataloaders(batch_size=32)
# model, loss_fn, optimizer = get_model_opt_loss()
model, loss_fn, optimizers = m.get_model_opt_loss(opt_name="adam", pretrain=False)

m.train(  # should this still be called train or do we name it def test?
    model=model,
    train_dataloader=train_dataloader,  # Since this is in the test-part, should we have test_dataloader?
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    optimizers=optimizers,
    # first_order_optimizer=first_order_optimizer,
    # second_order_optimizer=second_order_optimizer,
    epochs=10,
)
