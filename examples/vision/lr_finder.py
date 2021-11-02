# start_lr = 0.0001
# end_lr = 1.0
# num_batches = len(Data)/batch_size
# cur_lr = start_lr
# lr_multiplier = (end_lr / start_lr) ** (1.0 / num_batches)
# losses = []
# lrs = []for i in 1 to num_batches:
#     loss = train_model_get_loss(batch=i)
#     losses.append(loss)
#     lrs.append(cur_lr)
#     cur_lr = cur_lr*lr_multiplier # increase LRplot(lrs,losses)


def find_lr(init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in trn_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        # disable gradients?
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

logs,losses = find_lr()
plt.plot(logs[10:-5],losses[10:-5])

#perform on validation set or train set?

if suggest_lr:
    # 'steepest': the point with steepest gradient (minimal gradient)
    print("LR suggestion: steepest gradient")
    min_grad_idx = None
    try:
        min_grad_idx = (np.gradient(np.array(losses))).argmin()