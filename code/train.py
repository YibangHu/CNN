import torch
from torch import nn
from tqdm.auto import tqdm
def train(model, optimizer, loss_fn, train_dataloader, eval_dataloader, epochs, device):
    history = {} 
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    for epoch in tqdm(range(epochs)):
        #train
        model.train()
        train_loss = 0.0
        num_train_correct = 0
        num_train_examples = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * x.size(0)
            num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc = num_train_correct / num_train_examples
        train_loss = train_loss / len(train_dataloader.dataset)

        #evaluate
        model.eval()
        val_loss = 0.0
        num_val_correct = 0
        num_val_examples = 0

        for batch in eval_dataloader:

            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss += loss.data.item() * x.size(0)
            num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc = num_val_correct / num_val_examples
        val_loss = val_loss / len(eval_dataloader.dataset)

        if (epoch+1) == 1 or (epoch+1)%5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, train loss: {train_loss}, train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}')


        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
    return history