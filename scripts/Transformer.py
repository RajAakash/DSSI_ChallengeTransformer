import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib.backends.backend_pdf import PdfPages
from func_DL import CustomDataset  # Import your custom dataset handler

class TransformerModel(nn.Module):
    def __init__(self, input_features, model_dim, num_heads, num_layers, seq_length, output_dim=75):
        super(TransformerModel, self).__init__()
        self.input_embed = nn.Linear(input_features, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.input_embed(src)
        src += self.pos_encoder
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        return self.output_layer(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = TransformerModel(input_features=12, model_dim=512, num_heads=8, num_layers=3, seq_length=500, output_dim=75).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = CustomDataset(path_dir_X="../data_X", path_dir_Y="../data_Y_Task3", n_test=100, n_val=100, batch_size=100)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, target in dataloader:
        data = data.transpose(1, 2).to(device).float()
        target = target.view(-1, 75).to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        print(f'loss here {loss}')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

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

num_epochs = 1000
history = {"train_loss": [], "val_loss": []}

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plot and save to PDF
with PdfPages('training_validation_loss.pdf') as export_pdf:
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    export_pdf.savefig()
    plt.close()
