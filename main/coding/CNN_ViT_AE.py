import openpyxl
import torch.nn as nn
from decode import CustomDecoder
from VisionTransformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset
import torch
import xlrd
import numpy as np
import torch.optim as optim

class MultiScaleConvViT_AE(nn.Module):
    def __init__(self, num_classes):
        super(MultiScaleConvViT_AE, self).__init__()
        # Define multiple convolutional layers with different scales
        # First multi-scale convolutional layer
        self.basechannels=16
        self.conv1_3 = nn.Conv2d(1, self.basechannels, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv2d(1, self.basechannels, kernel_size=5, padding=2)
        self.conv1_7 = nn.Conv2d(1, self.basechannels, kernel_size=7, padding=3)

        # 2th multi-scale convolutional layer
        self.conv2_3 = nn.Conv2d(self.basechannels*3, self.basechannels*12, kernel_size=3, padding=1)  # 96 = 3 * 32
        self.conv2_5 = nn.Conv2d(self.basechannels*3, self.basechannels*12, kernel_size=5, padding=2)
        self.conv2_7 = nn.Conv2d(self.basechannels*3, self.basechannels*12, kernel_size=7, padding=3)

        # 3th multi-scale convolutional layer
        self.conv3 = nn.Conv2d(self.basechannels*36, 128, kernel_size=1, padding=1)  # 192 = 64 * 3

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)




        # ViT configuration parameters
        self.vit_1 = VisionTransformer(img_size=50, patch_size=10, num_classes=64, embed_dim=48, num_heads=4, ff_dim=256,num_layers=1, dropout_rate=0.3)  # vit_1
        self.vit_2 = VisionTransformer(img_size=7, patch_size=1, num_classes=64, embed_dim=128, num_heads=4, ff_dim=256, num_layers=1, dropout_rate=0.3) #vit_2

        self.fc = nn.Linear(64*2, num_classes)


        self.decode=CustomDecoder(num_classes)

    def forward(self, x):
        # x sahpe : (batch_size, 1, 50, 50)
        # Multi-scale convolution to extract features
        x=x.reshape(-1,1,50,50)
        # 1th multi-scale convolutional layer
        x1_3 = self.conv1_3(x)
        x1_5 = self.conv1_5(x)
        x1_7 = self.conv1_7(x)
        x1 = torch.cat((x1_3, x1_5, x1_7), dim=1)

        #Input to ViT
        vit_1_output = self.vit_1(x1)

        x1_pool_out = self.pool(nn.ReLU()(x1))
        # 2th multi-scale convolutional layer
        x2_3 = self.conv2_3(x1_pool_out)
        x2_5 = self.conv2_5(x1_pool_out)
        x2_7 = self.conv2_7(x1_pool_out)
        x2 = torch.cat((x2_3, x2_5, x2_7), dim=1)
        x2 = self.pool(nn.ReLU()(x2))

        # 3th multi-scale convolutional layer
        x3_pool_out = self.pool(nn.ReLU()(self.conv3(x2)))

        # Input to ViT
        vit_2_output = self.vit_2(x3_pool_out)

        vit_output=torch.cat((vit_1_output, vit_2_output), dim=1)

        # output is lvs
        output = self.fc(vit_output)  # Output, shape is (batch_size, num_dims)
        output=self.decode(output)

        return output

    def _get_position_embedding(self, seq_length, embed_dim):
        #  generating position embeddings
        position_embedding = torch.zeros(seq_length, embed_dim)
        for pos in range(seq_length):
            for i in range(0, embed_dim, 2):
                position_embedding[pos, i] = torch.sin(
                    torch.tensor(pos) / (10000 ** (torch.tensor(i) / embed_dim)))
                if i + 1 < embed_dim:
                    position_embedding[pos, i + 1] = torch.cos(
                        torch.tensor(pos) / (10000 ** (torch.tensor(i) / embed_dim)))
        return position_embedding.unsqueeze(0)

    def get_attention_weights(self):
        # Get attention weights from ViT (from each sub-model)
        vit_attention = self.vit.get_attention_maps()
        vitcnn_attention = self.vitcnn.get_attention_maps()
        return vit_attention, vitcnn_attention  # 返回两个模型的注意力权重



#-----------------------------------------------------vit_cnn_AE.py  train coding  ------------------------------------------------------------------------#


def normalized_data(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x-x_min)/(x_max-x_min)
    return x, x_max, x_min


def r_squared(x1, x2):
    # Calculate Total Sum of Squares (TSS)
    tss = np.sum((x2 - np.mean(x2)) ** 2)

    # Calculate Residual Sum of Squares (RSS)
    rss = np.sum((x2 - x1) ** 2)

    # Calculate R^2
    r2 = 1 - (rss / tss)

    return r2

def calculate_accuracy(x1, x2):
    x1=(x1>0.5).astype(int)
    x2=(x2>0.5).astype(int)
    acc = np.mean(np.abs(x1-x2))
    return 1-acc


def calculate_error(x1, x2):
    a = np.abs(x1-x2)
    b = x2
    e = np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0), axis=1)
    return e

def load_data(filename):
    book = xlrd.open_workbook(filename)
    sheet1 = book.sheet_by_name('images')  # Read data from the worksheet named 'images'
    m, n1 = sheet1.nrows, sheet1.ncols
    images = np.zeros((m, n1))


    # Convert data to 0 and 1
    for i in range(m):
        for j in range(n1):

            images[i, j] = 1 if sheet1.cell(i, j).value > 0.5 else 0

    # Here we divide the data into two groups: 0-249 and 249-end
    group1 = images[:249]  # 1th group of data
    group2 = images[249:]  # 2th group of data

    # Extract training, validation, and test sets from each group
    tr_images_1, va_images_1, te_images_1 = split_data(group1)
    tr_images_2, va_images_2, te_images_2 = split_data(group2)

    # Combine the results
    combined_tr_images = np.vstack((tr_images_1, tr_images_2))
    combined_va_images = np.vstack((va_images_1, va_images_2))
    combined_te_images = np.vstack((te_images_1, te_images_2))

    return combined_tr_images, combined_va_images, combined_te_images

def split_data(data):
    m = data.shape[0]
    tr_images = data[:int(m * 0.90)]  # 90% of data for training
    va_images = data[int(m * 0.90):int(m * 0.95)]  # 5% of data for training
    te_images = data[int(m * 0.95):] # 5% of data for training
    return tr_images, va_images, te_images



def training_model():
    # Load data

    tr_images, va_images, te_images = load_data("..\\dataset\\image.xlsx")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare training data

    train_inputs = torch.tensor(tr_images, dtype=torch.float32).to(device)
    train_targets = torch.tensor(tr_images, dtype=torch.float32).reshape(-1, 1, 50, 50).to(device)


    # Create TensorDataset and DataLoader for batching
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Adjust batch_size as needed

    # Prepare validation data
    va_inputs = torch.tensor(va_images, dtype=torch.float32).to(device)
    va_targets = torch.tensor(va_images, dtype=torch.float32).reshape(-1, 1, 50, 50).to(device)

    te_inputs = torch.tensor(te_images, dtype=torch.float32).to(device)
    te_targets = torch.tensor(te_images, dtype=torch.float32).reshape(-1, 1, 50, 50).to(device)


    # Define model
    model = MultiScaleConvViT_AE(4).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00002)

    tr_acc = []
    va_acc = []
    te_acc = []
    loss_history_train = []
    loss_history_validation = []

    num_epochs = 400  # Training epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_tr_accuracy=0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_tr_accuracy += calculate_accuracy(outputs.detach().cpu().numpy(), batch_targets.detach().cpu().numpy())


        loss_history_train.append(epoch_loss / len(train_loader))
        tr_accuracy=epoch_tr_accuracy/len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(va_inputs)
            val_loss = criterion(val_outputs, va_targets)

            te_outputs = model(te_inputs)
            te_loss = criterion(te_outputs, te_targets)

        loss_history_validation.append(val_loss.item())

        # Calculate accuracy

        va_accuracy = calculate_accuracy(val_outputs.detach().cpu().numpy(), va_targets.detach().cpu().numpy())
        te_accuracy = calculate_accuracy(te_outputs.detach().cpu().numpy(), te_targets.detach().cpu().numpy())

        tr_acc.append(tr_accuracy)
        va_acc.append(va_accuracy)
        te_acc.append(te_accuracy)

        if va_accuracy >= max(va_acc) and max(va_acc) > 0.97:
            torch.save(model.state_dict(),
                       "ViT.pth")  # Save the model state dict
            print("Model saved at epoch:", epoch)
            print(f"Training Accuracy: {tr_accuracy}, Validation Accuracy: {va_accuracy}, Test Accuracy: {te_accuracy}")

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss.item():.4f}, Test Loss: {te_loss.item():.4f},')
            print(f"Training Accuracy: {tr_accuracy}, Validation Accuracy: {va_accuracy}, Test Accuracy: {te_accuracy}")


    return loss_history_train, loss_history_validation, tr_acc, va_acc


if __name__ == "__main__":
    loss_train, loss_val, train_accuracy, val_accuracy = training_model()
