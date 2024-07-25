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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Transformer Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_features, model_dim, num_heads, num_layers, seq_length, output_dim=75):
        super(TransformerModel, self).__init__()
        self.input_embed = nn.Linear(input_features, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, model_dim))  # Changed to random
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)  # Increased layers
        self.output_layer = nn.Linear(model_dim, output_dim)
        self.norm = nn.LayerNorm(model_dim)  # Added normalization

    def forward(self, src):
        src = self.input_embed(src) + self.pos_encoder
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.norm(output)  # Normalize before final output
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

def plot_VmDatas_Task3_all(ActTime, title):
    import matplotlib.pyplot as plt
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
    plt.xlabel("Test data ID", fontsize=15)
    plt.ylabel("Number of Activation Map", fontsize=15)
    cbar.set_label('Activation Time', fontsize=15, rotation=270, labelpad=15)
    plt.show()

def plot_true_predict_from_y(y_predict, y_true, title, path_save=None):
    r = np.corrcoef(y_true, y_predict)[0][1]
    R2 = r2_score(y_true, y_predict)
    MAE = mean_absolute_error(y_true, y_predict)
    RMSE = np.sqrt(mean_squared_error(y_true, y_predict))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_predict, y_true, c="black", marker='.', linestyle="", ms=3)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.text(0.5, 0.94, f"$r$={r:.2f}, $R^2$={R2:.2f}, $MAE$={MAE:.2f}, $RMSE$={RMSE:.2f}", 
             fontsize=12, ha='center', transform=ax.transAxes)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    plt.title(title)
    if path_save:
        plt.savefig(path_save)
    plt.show()

# Main Execution Setup
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset(path_dir_X="../data_X", path_dir_Y="../data_Y_Task3", n_test=100, n_val=100, batch_size=100)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=100, shuffle=False)  # Assuming separate validation set or mechanism

    model = TransformerModel(input_features=12, model_dim=512, num_heads=8, num_layers=3, seq_length=500, output_dim=75).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 300
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)  # Now using separate val_loader
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if epoch == num_epochs - 1:  # Last epoch
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.transpose(1, 2).to(device).float()
                    target = target.view(-1, 75).to(device).float()
                    outputs = model(data)
                    final_outputs = outputs.to(device).detach().numpy()
                    final_targets = target.to(device).detach().numpy()

    with PdfPages('training_validation_loss1.pdf') as export_pdf:
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

        x_test, y_test = dataset.return_test_data()
        x_test = dataset.change_data_setting_to_train(x_test)
        y_test = dataset.change_data_setting_to_train(y_test)
        x_test = x_test.transpose(1, 2).to(device).float()
        y_test = y_test.view(-1, 75).to(device).float()
        outputs = model(x_test)
        outputs_np = outputs.to('cpu').detach().numpy().copy()
        y_test_np = y_test.to('cpu').detach().numpy().copy()

        plot_true_predict_from_y(outputs_np.flatten(), y_test_np.flatten(), "True vs Predicted at Final Epoch")
        export_pdf.savefig()
        plt.close()

    with PdfPages('Activation_Maps1.pdf') as export_pdf:
        output = outputs.to('cpu').view(75, dataset.return_n_test()).detach().numpy().copy() * 185
        y_test = y_test.to('cpu').view(75, dataset.return_n_test()).detach().numpy().copy() * 185
        plot_VmDatas_Task3_all(y_test, "True Activation Maps")
        export_pdf.savefig()
        plt.close()
        plot_VmDatas_Task3_all(output, "Predicted Activation Maps")
        export_pdf.savefig()
        plt.close()
