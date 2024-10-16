# 此脚本还需要进行进一步的适配，进行特征提取

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import matplotlib.pyplot as p

from src.utils import print_run_time

# 检查 CUDA 是否可用，并选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义超参数
input_size = 10     # 输入特征的维度
hidden_size = 20    # 隐藏层的特征数
output_size = 1     # 输出维度
num_layers = 2      # RNN 的层数
num_epochs = 50     # 训练的轮数
learning_rate = 0.01
sequence_length = 5 # 序列长度
batch_size = 16

# 人工生成的序列数据 (简单的随机数作为训练数据)
def generate_data(num_samples, input_size, sequence_length):
    X = torch.randn(num_samples, sequence_length, input_size)
    y = torch.randn(num_samples, output_size)
    return X, y

# 生成训练数据
num_samples = 10000
X_train, y_train = generate_data(num_samples, input_size, sequence_length)

# 将训练数据移到 GPU (如果可用)
X_train = X_train.to(device)
y_train = y_train.to(device)

# 定义一个简单的 RNN 模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播通过 RNN 层
        out, _ = self.rnn(x, h0)
        
        # 最后一层的输出传入全连接层
        out = self.fc(out[:, -1, :])
        return out

@print_run_time('RNN训练')
def test_rnn():
    # 实例化模型，并将模型移到 GPU
    model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        
        # 将数据分批处理
        for i in range(0, num_samples, batch_size):
            inputs = X_train[i:i+batch_size]
            targets = y_train[i:i+batch_size]
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 打印每个 epoch 的损失
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试模型
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, sequence_length, input_size).to(device)  # 生成一个随机输入
        test_output = model(test_input)
        print("Test Output:", test_output.cpu().numpy())

test_rnn()


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

if __name__ == "**src.models.networks.feature_extractor":
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
