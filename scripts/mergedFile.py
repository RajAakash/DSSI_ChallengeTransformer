import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import random_split
# from torchvision import models
%run func_DL.py
print(torch.cuda.is_available())

from torch.utils.data import Dataset, DataLoader

#---------------------------------------------
#var
path_dir_X = "../data_X"
path_dir_Y = "../data_Y_Task3"
n_test = 100
n_val = 100
batch_size = 100 #5000

#---------------------------------------------
#instance
dataset = CustomDataset(path_dir_X=path_dir_X, path_dir_Y=path_dir_Y, n_test=n_test, n_val=n_val, batch_size=batch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#----------------------------
#var (condition)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#----------------------------
#var (train)
num_epochs = 1000
n_print_train_result = 1
val_flag = True

#----------------------------
#init (model)
in_channels = dataset.return_shape_X()[0]
in_length = dataset.return_shape_X()[1]
out_channels = dataset.return_shape_Y()[0]
out_length = dataset.return_shape_Y()[1]
# model = model_task3(in_channels, in_length, out_channels, out_length, batch_size).to(device)
# #init model weight
# model.apply(init_normal_dist)
#----------------------------
#init (optimizer, scheduler)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,100,300,500], gamma=0.95)
#----------------------------
#init (loss_func)
#https://neptune.ai/blog/pytorch-loss-functions
loss_func = nn.MSELoss()
#loss_func = nn.L1Loss()

import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_features, model_dim, num_heads, num_layers, seq_length, output_dim=75):
        super(TransformerModel, self).__init__()
        self.input_embed = nn.Linear(input_features, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        # print("Input src shape:", src.shape)
        src = self.input_embed(src)
        # print("After embedding shape:", src.shape)
        src += self.pos_encoder
        # print("After pos_encoder shape:", src.shape)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        return self.output_layer(output)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_features=12, model_dim=512, num_heads=8, num_layers=3, seq_length=500, output_dim=75).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, target in dataloader:
        data = data.transpose(1, 2).to(device).float()
        target = target.view(-1, 75).to(device).float()
        # print(f'target shape: {target.shape}')
        optimizer.zero_grad()
        output = model(data)
        # print(f' output shape: {output.shape}')
        loss = criterion(output, target)
        print(f'Loss here {loss}')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss = train_epoch(model, dataloader, optimizer, criterion, device)
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
    
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.transpose(1, 2).to(device).float()
            target = target.view(-1, 75).to(device).float()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Assume validation set is part of your dataset, if not, remove validation components
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

history = {"train_loss": [], "val_loss": []}
num_epochs = 2
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate_epoch(model, val_loader, criterion, device) if val_flag else None
    history["train_loss"].append(train_loss)
    if val_flag:
        history["val_loss"].append(val_loss)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}' + (f', Val Loss: {val_loss:.4f}' if val_flag else ''))

import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
if val_flag:
    plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print(history)

#----------------------------
# Assuming history["train_loss"] and history["val_loss"] are lists of floats
train_loss_tensor = torch.tensor(history["train_loss"])  # Convert list of floats to a tensor
val_loss_tensor = torch.tensor(history["val_loss"])      # Convert list of floats to a tensor

# No need to use torch.stack() if you're already creating tensors from lists directly
train_loss_np = train_loss_tensor.to('cpu').detach().numpy()
val_loss_np = val_loss_tensor.to('cpu').detach().numpy()

# Plotting the losses
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1,1,1)
ax.plot(train_loss_np, color="black", label="Train")
ax.plot(val_loss_np, color="maroon", label="Validation")
ax.tick_params(labelsize=20)
ax.set_xlabel("Epoch", fontsize=30)
ax.set_ylabel("MSE", fontsize=30)
ax.legend(fontsize=25, frameon=False)
ax.set_ylim(0, max(max(train_loss_np), max(val_loss_np)) * 1.1)  # Set the upper limit to 10% more than the max value
plt.show()

from matplotlib.backends.backend_pdf import PdfPages
model.eval()

with torch.no_grad():
    # Get test data
    x, y = dataset.return_test_data()  # Ensure this method exists and returns properly formatted tensors
    
    # Change the data setting if necessary (not always required)
    x = dataset.change_data_setting_to_train(x)  # Modify or ensure this method fits your data processing
    y = dataset.change_data_setting_to_train(y)
    print(x.shape)
    # Transfer tensors to the correct device
    # x = x.to(device)
    # y = y.to(device)
    x = x.transpose(1, 2).to(device).float()
    y = y.view(-1, 75).to(device).float()
    
    # Forward pass to get output/logits
    outputs = model(x)
    
    # Convert outputs to numpy for evaluation with sklearn
    outputs_np = outputs.to('cpu').detach().numpy().flatten()
    y_np = y.to('cpu').detach().numpy().flatten()
    
    # Compute metrics
    loss_MSE = mean_squared_error(y_np, outputs_np)
    loss_R2 = r2_score(y_np, outputs_np)

    # Print the losses
    print("MSE: ", loss_MSE)
    print("R2: ", loss_R2)

    # Plot the true vs. predicted values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_np, outputs_np, alpha=0.5)
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'k--', lw=4)  # Diagonal line
    plt.show()

    # Save to PDF if needed
    with PdfPages('model_evaluation.pdf') as export_pdf:
        plt.figure()
        plt.scatter(y_np, outputs_np, alpha=0.5)
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'k--', lw=4)  # Diagonal line
        export_pdf.savefig()
        plt.close()
        
from sklearn.metrics import r2_score # スコア計算
from sklearn.metrics import mean_absolute_error # スコア計算 (MAE)
from sklearn.metrics import mean_squared_error # スコア計算 (MSE)
#----------------
def plot_true_predict_from_y(y_predict_list:pd.DataFrame, y_true_list:pd.DataFrame,
                            title:str, path_save=False) -> None:
    #----------------
    #calc score
    r = np.corrcoef(y_true_list, y_predict_list)[0][1]
    R2 = r2_score(y_true=y_true_list, y_pred=y_predict_list) # 決定係数(R2) #https://bellcurve.jp/statistics/course/9706.html
    MAE = mean_absolute_error(y_true=y_true_list, y_pred=y_predict_list) # 平均絶対誤差(MAE)
    RMSE = np.sqrt(mean_squared_error(y_true=y_true_list, y_pred=y_predict_list)) # 二乗平均平方根誤差(RMSE)
    
    #----------------
    #fig, ax
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)
    #----------------
    #plot scatter
    #ax.scatter(x=y_predict_list, y=y_true_list, s=40, c="black", marker="o", zorder=10)
    ax.plot(y_predict_list, y_true_list, c="black", marker='.', linestyle="", ms=3, zorder=10)
    #----------------
    #plot 直線
    x=np.linspace( min(min(y_true_list),min(y_predict_list)), max(max(y_true_list),max(y_predict_list)), 10) #listの足し算は結合
    y=x
    ax.plot(x, y, color = "black")
    #----------------
    #plot text
    plt.text(x=0.5, y=0.94, 
             s="$r$={0}, $R^2$={1}, $MAE$={2}, $RMSE$={3}".format("{:.2f}".format(r),
                                                                 "{:.2f}".format(R2),
                                                                "{:.2f}".format(MAE),
                                                                "{:.2f}".format(RMSE)), 
             fontdict=dict(fontsize=25, color="black"), ha='center', transform=ax.transAxes,
             zorder=20)
    #----------------
    #setting
    ax.tick_params(labelsize = 20)#軸の大きさ
    ax.set_xlabel("True",fontsize=30)
    ax.set_ylabel("Predict",fontsize=30)
    plt.title("{0}".format(title), fontsize=30)
    #----------------
    #save
    if path_save != False:
        plt.savefig(path_save, bbox_inches='tight')
    #----------------
    #show
    plt.show()
    
#----------------------------
# eval
model.eval()
with torch.no_grad():
    #----------------------------
    #forward
    x, y = dataset.return_test_data()
    #----------------------------
    #float32, grad==True
    x = dataset.change_data_setting_to_train(x)
    y = dataset.change_data_setting_to_train(y)
    #----------------------------
    #change the type
    # x = x.to(device)
    # y = y.to(device)
    x = x.transpose(1, 2).to(device).float()
    y = y.view(-1, 75).to(device).float()
    #----------------------------
    #forward
    output = model(x)
    #----------------------------
    #change to numpy
    output = output.to('cpu').detach().numpy().copy().flatten() * 185
    y = y.to('cpu').detach().numpy().copy().flatten() * 185
    #----------------------------
    #plot
    plot_true_predict_from_y(y_predict_list=output, y_true_list=y, title="", path_save=False) 
    
def plot_VmDatas_Task3_all(ActTime, title):
    # plot the Activation Time array
    plt.imshow(ActTime, cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Activation Time')
    cbar = plt.colorbar()
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    # not xticks
    #plt.xticks([])
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title("{0}".format(title), fontsize=20)
    plt.xlabel("Test data ID",fontsize=15)
    plt.ylabel("Number of Activation Map",fontsize=15)
    cbar.set_label('Activation Time', fontsize=15, rotation=270, labelpad=15)
    plt.show()
    
#var
n_plot = 3

#----------------------------
# eval
model.eval()
with torch.no_grad():
    #----------------------------
    #forward
    x, y = dataset.return_test_data()
    #----------------------------
    #float32, grad==True
    x = dataset.change_data_setting_to_train(x)
    y = dataset.change_data_setting_to_train(y)
    #----------------------------
    #change the type
    # x = x.to(device)
    # y = y.to(device)
    x = x.transpose(1, 2).to(device).float()
    y = y.view(-1, 75).to(device).float()
    #----------------------------
    #forward
    output = model(x)
    #----------------------------
    #change to numpy
    output = output.to('cpu').view(75, dataset.return_n_test()).detach().numpy().copy() * 185
    y = y.to('cpu').view(75, dataset.return_n_test()).detach().numpy().copy() * 185
    #----------------------------
    #plot
    plot_VmDatas_Task3_all(y, title="True")
    plot_VmDatas_Task3_all(output, title="Predict")