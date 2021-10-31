# MODEL EMA EXAMPLE

# train_dataset, eval_dataset = create_datasets()
#     train_dataloader, eval_dataloader = create_dataloaders()
#     model = create_model()
#
#     # wrap model in ModelEMA class
#     ema_model = ModelEMA(model)
#
#     for epoch in EPOCHS:
#         batch = next(iter(train_dataloader)
#         loss = model(** batch)
#         loss.backward()
#
#         # apply EMA to model weights
#         ema_model.update(model)


# label smoothing - timm

# drop path - timm

# dropout

# rand augment - timm

# optimizer SGDR (warm restarts) + Cosine scheduling
# or LAMB cosine scheduler

# architecture - resnet RS

# loss - BCE loss

#mixup and cutmix

# repeated augmentation + stochastic depth : tend to improve the results at convergence, but they slow
# down the training in the early stages


