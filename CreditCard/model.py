import torch.nn as nn
import torch.optim as optim
import tqdm
from torchmetrics.classification import BinaryF1Score
import copy
import torch
import numpy as np

class BinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(20, 60)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.sigmoid(self.output(x))
        return x

    def model_train(self,model, X_train, y_train, X_val, y_val, weight_value,writer,fold,weights = True):
        # loss function and optimizer
        if weights == True:
          weight = torch.tensor([weight_value])
          loss_fn = nn.BCELoss(weight = weight)  # binary cross entropy
        else:
          loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
        # print("here")
        n_epochs = 10000   # number of epochs to run
        batch_size = 500  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)
        metric = BinaryF1Score()
        # Hold the best model
        best_acc = - np.inf   # init to negative infinity
        best_weights = None
        # print(y_train)
        for epoch in range(n_epochs):
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # print(X_batch.shape,y_batch.shape)
                    # forward pass
                    y_pred = model(X_batch)
                    # print("here")
                    loss = loss_fn(y_pred.view(-1, 1).float(), y_batch)
                    # print("here")
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    acc = (y_pred.round() == y_batch).float().mean()
                    f1 = metric(y_pred, y_batch)
            writer.add_scalar(f'loss/train/fold{fold}/weight{weight_value}', loss, epoch)
            writer.add_scalar(f'acc/train/fold{fold}/weight{weight_value}', acc, epoch)
            writer.add_scalar(f'f1/train/fold{fold}/weight{weight_value}', f1, epoch)

            # print("Training Accuracy:",acc)
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            acc = float(acc)
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
            
            loss_pred = loss_fn(y_pred.view(-1, 1).float(), y_val)
            acc_pred = acc
            f1_pred =  metric(y_pred, y_val)
            writer.add_scalar(f'loss/test/fold{fold}/weight{weight_value}', loss_pred, epoch)
            writer.add_scalar(f'acc/test/fold{fold}/weight{weight_value}', acc_pred, epoch)
            writer.add_scalar(f'f1/test/fold{fold}/weight{weight_value}', f1_pred, epoch)
        # restore model and return best accuracy
        # f1 = metric(y_pred, y_val)
        model.load_state_dict(best_weights)
        return best_acc, f1