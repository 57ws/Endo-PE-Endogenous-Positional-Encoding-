import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, r2_score
import time
import copy

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 数据生成
class DistanceDataset(Dataset):
    def __init__(self, num_samples=10000, seq_length=100):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        self._generate_data()

    def _generate_data(self):
        for _ in range(self.num_samples):
            # 创建序列
            seq = np.random.randint(0, 9, self.seq_length)

            # 随机选择两个不同位置设为9
            pos1, pos2 = np.random.choice(self.seq_length, 2, replace=False)
            seq[pos1] = 9
            seq[pos2] = 9

            # 计算距离
            distance = abs(pos1 - pos2)

            self.data.append(seq)
            self.labels.append(distance)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


# 创建数据集
dataset = DistanceDataset(num_samples=14000)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# 数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 经典正弦位置编码
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Control(nn.Module):
    def __init__(self, d_model, max_len=5000, nhead=4, dropout=0.1, sin=True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead
        self.sin = sin

        self.sin_encoder = SinusoidalPositionalEncoding(d_model, max_len)

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        batch_size, seq_len, _ = x.shape

        if self.sin:
            x = self.sin_encoder(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x
        )

        # 残差连接和归一化
        x = self.norm1(x)

        return x + res


class GRUPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        dense_out = self.dense(gru_out)
        return x + dense_out


# 以逻辑掩码为核心的位置编码
class LogicalMaskEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # 归一化层
        self.norm = nn.LayerNorm(d_model)

        self.relu = nn.ReLU()

        # 创建逻辑掩码（上三角矩阵）
        self.register_buffer("mask", self.generate_mask(max_len))

    def generate_mask(self, size):
        """创建上三角掩码矩阵"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        res = x

        batch_size, seq_len, _ = x.shape

        # 应用带掩码的自注意力
        attn_mask = self.mask[:seq_len, :seq_len]
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask
        )

        # 归一化
        x = self.norm(x)

        x = self.relu(x)

        return res + x


# 正弦与逻辑掩码混合
class sin_LogicalMskEncoding_mask(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 基础正弦位置编码
        self.sin_encoder = SinusoidalPositionalEncoding(d_model, max_len)

        # 自注意力层
        self.self_attn = LogicalMaskEncoding(d_model, max_len)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)

        x = self.sin_encoder(x)

        x = self.self_attn(x)

        return x


# 正弦与gru混合
class sin_Gru(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 基础正弦位置编码
        self.sin_encoder = SinusoidalPositionalEncoding(d_model, max_len)

        # gru
        self.self_gru = GRUPositionalEncoding(d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)

        x = self.self_gru(x)

        x = self.sin_encoder(x)

        return x


# 3. 主模型
class DistancePredictor(nn.Module):
    def __init__(self, pe_type='sinusoidal', vocab_size=10, d_model=128, nhead=4,
                 dim_feedforward=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码选择
        if pe_type == 'sinusoidal':
            self.positional_encoding = Control(d_model, sin=True)
        elif pe_type == 'gru':
            self.positional_encoding = GRUPositionalEncoding(d_model)
        elif pe_type == 'Logical':
            self.positional_encoding = LogicalMaskEncoding(d_model)
        elif pe_type == 'sin_gru':
            self.positional_encoding = sin_Gru(d_model)
        elif pe_type == 'sin_Logical':
            self.positional_encoding = sin_LogicalMskEncoding_mask(d_model)
        elif pe_type == 'none':  # 无位置编码
            self.positional_encoding = Control(d_model, sin=False)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 回归头
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 嵌入层
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # 位置编码
        x = self.positional_encoding(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 池化和回归
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        x = self.fc(x).squeeze(-1)  # (batch)
        return x


# 4. 训练和评估函数（添加早停机制）
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=5):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # 检查验证损失是否改善
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        # 早停检查
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, '
              f'Best Val Loss: {best_val_loss:.4f}, No Improve: {epochs_no_improve}/{patience}')

    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    return train_losses, val_losses


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = total_loss / len(test_loader.dataset)

    # 计算回归指标
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f'Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}')
    return test_loss, mse, r2, all_preds, all_labels


# 5. 可视化函数
def plot_loss_curves(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    # plt.savefig(f'{model_name}_loss_curve.png')
    plt.show()


def plot_predictions_vs_actual(all_labels, all_preds, model_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([0, 100], [0, 100], 'r--')  # 理想预测线
    plt.xlabel('Actual Distance')
    plt.ylabel('Predicted Distance')
    plt.title(f'{model_name} Predictions vs Actual')
    plt.grid(True)
    # plt.savefig(f'{model_name}_predictions.png')
    plt.show()


def plot_error_distribution(all_labels, all_preds, model_name):
    errors = np.array(all_labels) - np.array(all_preds)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} Prediction Error Distribution')
    plt.grid(True)
    # plt.savefig(f'{model_name}_error_dist.png')
    plt.show()


# 6. 实验运行
pe_types = [
    'none',
    'sinusoidal',
    'gru',
    'sin_gru',
    'Logical',
    'sin_Logical'
]
results = {}
all_metrics = []

for pe_type in pe_types:
    print(f"\n{'=' * 50}")
    print(f"Training model with {pe_type if pe_type != 'none' else 'no'} positional encoding")
    print(f"{'=' * 50}")

    # 创建模型
    model = DistancePredictor(pe_type=pe_type).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    start_time = time.time()
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion,
        optimizer, num_epochs=20, patience=5
    )
    training_time = time.time() - start_time

    # 评估模型
    test_loss, mse, r2, preds, labels = evaluate_model(
        model, test_loader, criterion
    )

    # 存储结果
    results[pe_type] = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'mse': mse,
        'r2': r2,
        'preds': preds,
        'labels': labels,
        'training_time': training_time
    }

    # 可视化
    # plot_loss_curves(train_losses, val_losses, f"{pe_type}_pe")
    # plot_predictions_vs_actual(labels, preds, f"{pe_type}_pe")
    # plot_error_distribution(labels, preds, f"{pe_type}_pe")

    # 记录指标
    all_metrics.append({
        'Positional Encoding': pe_type,
        'Test MSE': mse,
        'R² Score': r2,
        'Training Time (s)': training_time
    })

# 7. 结果比较
print("\nFinal Comparison of All Models:")
print("{:<20} {:<10} {:<10} {:<15}".format(
    'Positional Encoding', 'MSE', 'R²', 'Training Time (s)'))
for metrics in all_metrics:
    print("{:<20} {:<10.4f} {:<10.4f} {:<15.2f}".format(
        metrics['Positional Encoding'],
        metrics['Test MSE'],
        metrics['R² Score'],
        metrics['Training Time (s)']))

# 绘制所有模型的损失曲线对比
plt.figure(figsize=(12, 8))
for pe_type in pe_types:
    if pe_type in results:
        plt.plot(results[pe_type]['val_losses'], label=f"{pe_type} PE")
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)
# plt.savefig('all_models_val_loss.png')
plt.show()