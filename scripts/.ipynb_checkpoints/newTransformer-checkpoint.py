# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from func_DL import CustomDataset  # Assuming func_DL.py has the CustomDataset class
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages

# Transformer Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_features, model_dim, num_heads, num_layers, seq_length, output_dim=75):
        super(TransformerModel, self).__init__()
        self.input_embed = nn.Linear(input_features, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.input_embed(src) + self.pos_encoder
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        return self.output_layer(output)

# Training and Validation Functions
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, target in dataloader:
        data = data.transpose(1, 2).to(device).float()
        target = target.view(-1, 75).to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
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

# Main Execution Setup
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = CustomDataset(path_dir_X="../data_X", path_dir_Y="../data_Y_Task3", n_test=100, n_val=100, batch_size=100)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)
    model = TransformerModel(input_features=12, model_dim=512, num_heads=8, num_layers=3, seq_length=500, output_dim=75).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, train_loader, criterion, device)  # Assuming same loader for simplicity
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Saving plots as PDF
    with PdfPages('training_validation_loss.pdf') as export_pdf:
        plt.figure()
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
