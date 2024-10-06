import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_encoder_layers=2):
        super(TransformerRegressor, self).__init__()
        
        # Embedding layer to project input data to model dimension
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), 
            num_layers=num_encoder_layers
        )
        
        # Output regression layer
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Embedding input
        x = self.embedding(x)
        
        # Apply transformer encoder (we assume no mask or src_key_padding_mask for simplicity)
        x = self.transformer_encoder(x)
        
        # Take the mean over sequence length dimension (batch_size, seq_len, d_model) -> (batch_size, d_model)
        x = x.mean(dim=1)
        
        # Output 2D regression value
        output = self.fc_out(x)
        
        return output

if __name__ == "__main__":
    # 生成随机数据
    np.random.seed(42)
    torch.manual_seed(42)

    # 假设有1000个样本，每个样本有40维输入特征，输出是2维
    X = np.random.rand(1000, 10, 40).astype(np.float32)  # shape: (1000, 10, 40)
    y = np.random.rand(1000, 2).astype(np.float32)       # shape: (1000, 2)

    # 将数据转换为 PyTorch 张量
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    # 使用 DataLoader 和 Dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerRegressor(input_dim=40, output_dim=2).to(device)
    criterion = nn.MSELoss()  # 回归问题的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和验证的次数
    epochs = 50

    # 用于保存训练和验证损失
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_model(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()



# 定义训练函数
def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(data_loader)

# 定义验证函数
def validate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()

    return total_loss / len(data_loader)
