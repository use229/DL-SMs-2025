import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pytorch_VAE_Decoder import CustomDecoder
from pytorch_VAE_Encoder import Encoder
import time
class VAE(nn.Module):
    def __init__(self, n_z):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_z)
        self.decoder = CustomDecoder(n_z)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z), z_mean, z_log_var


def calculate_accuracy(x1, x2):
    x1=(x1>0.5).astype(int)
    x2=(x2>0.5).astype(int)
    acc = np.mean(np.abs(x1-x2))
    return 1-acc
# Load Data Function
def load_data_2000(filename):
    df = pd.read_excel(filename, sheet_name='images')
    lvs = (df.values > 0.5).astype(np.float32)  # Convert to binary
    group1 = lvs[:249]  # First group
    group2 = lvs[249:]  # Second group

    tr_lvs_1, va_lvs_1 = split_data(group1)
    tr_lvs_2, va_lvs_2 = split_data(group2)

    combined_tr_lvs = np.vstack((tr_lvs_1, tr_lvs_2))
    combined_va_lvs = np.vstack((va_lvs_1, va_lvs_2))

    return combined_tr_lvs, combined_va_lvs

def split_data(data):
    m = data.shape[0]
    tr_lvs = data[:int(m * 0.90)]
    va_lvs = data[int(m * 0.90):]
    return tr_lvs, va_lvs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Num GPUs Available: ", torch.cuda.device_count())

    tr_x, va_x = load_data_2000("..\\..\\dataset\\image.xlsx")
    tr_x = torch.tensor(tr_x).view(-1, 1, 50, 50).to(device)
    va_x = torch.tensor(va_x).view(-1, 1, 50, 50).to(device)

    model = VAE(n_z=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss(reduction='sum')
    tr_acc = []
    va_acc=[]
    for epoch in range(100000):
        model.train()
        start_time = time.time()
        optimizer.zero_grad()
        recon_batch, z_mean, z_log_var = model(tr_x)
        recon_loss = criterion(recon_batch, tr_x)
        latent_loss = -0.5 * torch.mean(z_log_var - z_mean.pow(2) - z_log_var.exp() + 1)
        loss = (recon_loss + latent_loss)/tr_x.shape[0]
        loss.backward()
        optimizer.step()
        print(epoch, 'Train Loss:', loss.item())
        if epoch%50==0:
            tr_accuracy = calculate_accuracy(recon_batch.detach().cpu().numpy(), tr_x.detach().cpu().numpy())
            tr_acc.append(tr_accuracy)
            print('calculate_accuracy:', tr_accuracy)


        # Validation loss
        model.eval()
        with torch.no_grad():
            recon_batch, z_mean, z_log_var = model(va_x)
            recon_loss = criterion(recon_batch, va_x)
            latent_loss = -0.5 * torch.mean(z_log_var - z_mean.pow(2) - z_log_var.exp() + 1)
            val_loss = (recon_loss + latent_loss)/va_x.shape[0]
            print('Validation Loss:', val_loss.item())
            va_accuracy = calculate_accuracy(recon_batch.detach().cpu().numpy(), va_x.detach().cpu().numpy())
            va_acc.append(va_accuracy)
            print('calculate_accuracy:', va_accuracy)
            torch.save(model.state_dict(), f"..\\..\\newMoudle\\VAE\\test")
            if va_accuracy >= max(va_acc) and max(va_acc) > 0.95:
                torch.save(model.state_dict(),f"..\\..\\newMoudle\\VAE\\vae_simple_{va_acc}")  # Save the model state dict
                print("Model saved at epoch:", epoch)
                print(f"Training Accuracy: {tr_accuracy}, Validation Accuracy: {va_accuracy}")
            if(epoch%1000==0 and epoch>0):
                torch.save(model.state_dict(),f"..\\..\\newMoudle\\VAE\\vae_simple{epoch}")  # Save the model state dict
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training Time: {training_time:.6f} seconds")

