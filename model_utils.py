import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import copy

def weighted_MSELoss(weights = [], device = "cpu"):
    '''
    Plots the distribution of the given features across the given samples
    -----------------------------
    :params:
        list weights: relative weights to be placed on each output timestep
        str device: hardware device to be used for computation
    -----------------------------
    :returns:
        function: configurated weighted mean square error loss function
    '''
    def f(output, target):
        if len(weights) == 0:
            return nn.MSELoss()(output, target)
        total = torch.zeros(1).to(device) # [0]
        for i in range(len(weights)):
            total += nn.MSELoss()(output[:,:,i], target[:,:,i]) * weights[i]
        return total / sum(weights)
    return f

def train_epoch(adj_mat, dataloader, model, optimizer, criterion, device):
    '''
    Trains the model for one epoch
    -----------------------------
    :params:
        list (2 dimensions of roads, roads) adj_mat: adjacency matrix (undirected or directed) for road network
        DataLoader dataloader: dataloader object to yield training data and targets
        Module model: pytorch model to train
        Optimizer optimizer: optimizer to be used for training
        Module criterion: loss function for training
        str device: hardware device to be used for computation
    -----------------------------
    :returns:
        float: mean epoch loss (averaged over batches)
    '''
    adj_mat = adj_mat.to(device)
    model.to(device)
    model.train()
    epoch_losses = []
    for batch_input, batch_target in tqdm(dataloader):
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad()
        output = model(adj_mat, batch_input) if adj_mat != None else model(batch_input)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        epoch_losses.append(batch_loss)
    return sum(epoch_losses) / len(epoch_losses)

def validate(adj_mat, dataloader, model, criterion, device):
    '''
    Validates the model
    -----------------------------
    :params:
        list (2 dimensions of roads, roads) adj_mat: adjacency matrix (undirected or directed) for road network
        DataLoader dataloader: dataloader object to yield validation data and targets
        Module model: pytorch model to validate
        Module criterion: loss function for validation
        str device: hardware device to be used for computation
    -----------------------------
    :returns:
        float: mean validation loss (averaged over batches)
    '''
    adj_mat = adj_mat.to(device)
    model.to(device)
    val_losses = []
    with torch.no_grad():
        model.eval()
        for batch_input, batch_target in dataloader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            output = model(adj_mat, batch_input) if adj_mat != None else model(batch_input)
            loss = criterion(output, batch_target)
            val_losses.append(loss.item())
    return sum(val_losses) / len(val_losses)

def train(adj_mat, train_loader, val_loader, model, optimizer, criterion, patience, epochs, device):
    '''
    Trains the model and plots the loss curve
    -----------------------------
    :params:
        list (2 dimensions of roads, roads) adj_mat: adjacency matrix (undirected or directed) for road network
        DataLoader train_loader: dataloader object to yield training data and targets
        DataLoader val_loader: dataloader object to yield validation data and targets
        Module model: pytorch model to train
        Optimizer optimizer: optimizer to be used for training
        Module criterion: loss function for training
        int patience: patience for early stopping
        int epochs: maximum number of epochs to train the model for
        str device: hardware device to be used for computation
    -----------------------------
    :returns:
        None
    '''
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
    
def predict(adj_mat, dataloader, model, means, stds, device, repeated = False):
    '''
    Generates predictions from model
    -----------------------------
    :params:
        list (2 dimensions of roads, roads) adj_mat: adjacency matrix (undirected or directed) for road network
        DataLoader dataloader: dataloader object to prediction input data and targets
        Module model: pytorch model
        list means: means of the input features to the model
        list stds: standard deviations of the input features to the model
        str device: hardware device to be used for computation
        bool repeated: whether to do recursive prediction
    -----------------------------
    :returns:
        list (3 dimensions of samples, roads, output timesteps): predictions
        list (3 dimensions of samples, roads, output timesteps): actual targets
    '''
    adj_mat = adj_mat.to(device)
    model.to(device)
    preds = []
    actuals = []
    with torch.no_grad():
        model.eval()
        for batch_input, actual in tqdm(dataloader):
            actual = actual.cpu()
            actual = actual * stds[0] + means[0]
            actuals.append(actual)
            if repeated:
                pred = predict_repeated(adj_mat, batch_input, dataloader.dataset.num_timesteps_output, model, means, stds, device)
            else:
                batch_input = batch_input.to(device)
                pred = model(adj_mat, batch_input) if adj_mat != None else model(batch_input)
                pred = pred.cpu()
            pred = pred * stds[0] + means[0]
            preds.append(pred)
    
        return torch.cat(preds), torch.cat(actuals)
    
def predict_repeated(adj_mat, batch_input, num_timesteps_output, model, means, stds, device):
    '''
    Generates recursive predictions from model
    -----------------------------
    :params:
        list (2 dimensions of roads, roads) adj_mat: adjacency matrix (undirected or directed) for road network
        list batch_input: input batch from dataloader
        int num_timesteps_output: number of output timesteps
        Module model: pytorch model
        list means: means of the input features to the model
        list stds: standard deviations of the input features to the model
        str device: hardware device to be used for computation
    -----------------------------
    :returns:
        list (3 dimensions of samples, roads, output timesteps): predictions
    '''
    preds = [] # final predictions
    for i in range(num_timesteps_output):
        pred = model(adj_mat, batch_input.to(device)).cpu() if adj_mat != None else model(batch_input.to(device)).cpu()
        preds.append(pred)
        pred = torch.unsqueeze(pred, pred.dim())
        pred = torch.cat((pred, batch_input[:,:,0:1,1:]), pred.dim()-1) # adding missing features of output from input
        pred = pred * stds + means
        for i in range(pred.shape[0]): # adjusting the time features
            day = pred[i,0,0,3].item()
            hour = pred[i,0,0,4].item() + 1
            if hour == 25: # next day
                day += 1
                hour = 0
            pred[i,:,:,3] = day
            pred[i,:,:,4] = hour
        pred = (pred - means) / stds
        batch_input = torch.cat((batch_input, pred), 2)
        batch_input = batch_input[:,:,1:,:]
    return torch.cat(preds, dim=2)
    
def save_model(model, model_path):
    '''
    Saves the model in a model file
    -----------------------------
    :params:
        Module model: pytorch model to save
        str model_path: path to model file
    -----------------------------
    :returns:
        None
    '''
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    '''
    Loads a model file into the given model
    -----------------------------
    :params:
        Module model: pytorch model to load into
        str model_path: path to model file
    -----------------------------
    :returns:
        None
    '''
    model.load_state_dict(torch.load(model_path))