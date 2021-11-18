import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import copy

def weighted_MSELoss(weights = [], device = "cpu"):
    def f(output, target):
        if len(weights) == 0:
            return nn.MSELoss()(output, target)
        total = torch.zeros(1).to(device) # [0]
        for i in range(len(weights)):
            total += nn.MSELoss()(output[:,:,i], target[:,:,i]) * weights[i]
        return total / sum(weights)
    return f

def train_epoch(adj_mat, dataloader, model, optimizer, criterion, device):
    model.to(device)
    model.train()
    epoch_losses = []
    for batch_input, batch_target in dataloader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad()
        output = model(adj_mat, batch_input)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        epoch_losses.append(batch_loss)
    return sum(epoch_losses) / len(epoch_losses)

def validate(adj_mat, dataloader, model, criterion, device):
    model.to(device)
    val_losses = []
    with torch.no_grad():
        model.eval()
        for batch_input, batch_target in dataloader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            output = model(adj_mat, batch_input)
            loss = criterion(output, batch_target)
            val_losses.append(loss.item())
    return sum(val_losses) / len(val_losses)

def train(adj_mat, train_loader, val_loader, model, optimizer, criterion, patience, epochs, device):
    train_losses = []
    val_losses = []
    curr_patience = patience
    lowest_val_loss = float("inf")
    best_model_state = model.state_dict()
    for i in tqdm(range(epochs)):
        train_losses.append(train_epoch(adj_mat, train_loader, model, optimizer, criterion, device))
        val_losses.append(validate(adj_mat, val_loader, model, criterion, device))
        if val_losses[-1] < lowest_val_loss:
            lowest_val_loss = val_losses[-1]
            best_model_state = copy.deepcopy(model.state_dict())
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience == 0:
                print("Early stopping activated after epoch {}".format(i+1))
                break
    model.load_state_dict(best_model_state)
    
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.show()
    
def predict(adj_mat, dataloader, model, means, stds, device):
    preds = torch.empty((0, dataloader.dataset.num_nodes, dataloader.dataset.num_timesteps_output))
    actuals = torch.empty((0, dataloader.dataset.num_nodes, dataloader.dataset.num_timesteps_output))
    with torch.no_grad():
        model.eval()
        for batch_input, actual in tqdm(dataloader):
            batch_input = batch_input.to(device)
            actual = actual.cpu()
            actual = actual * stds[0] + means[0]
            actuals = torch.cat((actuals, actual))
            pred = model(adj_mat, batch_input).cpu()
            pred = pred * stds[0] + means[0]
            preds = torch.cat((preds, pred))
        return preds, actuals
    
def save_model(model, model_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))