import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import scipy.io
import random

# 固定随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== KULAAD脑电数据集加载器 ====================
class EEGData_KULAAD:
    def __init__(self, data_root, batch_size=32, window_points=128):
        self.data_root = data_root
        self.batch_size = batch_size
        self.window_points = window_points
        self.label_map = {'R': 0, 'L': 1}

    def load_single_subject(self, subject_index):
        mat_file = f"AAD_DataSet_S{subject_index}.mat"
        mat_path = os.path.join(self.data_root, mat_file)

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"文件不存在: {mat_path}")

        mat_data = scipy.io.loadmat(mat_path)
        aad_dataset = mat_data['AAD_DataSet']

        eeg_samples, ear_labels = self._process_eeg_samples(aad_dataset)
        X = torch.FloatTensor(eeg_samples)  # [num_windows, window_points, channels]
        y = torch.LongTensor([self.label_map[label] for label in ear_labels])
        return X, y

    def _process_eeg_samples(self, aad_dataset, num_trials=8):
        eeg_samples = []
        ear_labels = []

        for trial_idx in range(num_trials):
            # 获取EEG数据和注意方向
            eeg_data = aad_dataset['EEG20trials'][0][0][0][trial_idx]
            attention_ear = aad_dataset['Direction_Attention20trials'][0][0][0][trial_idx][0]

            # 分割时间窗口
            num_points = eeg_data.shape[0]
            num_frames = num_points // self.window_points

            for frame_idx in range(num_frames):
                start = frame_idx * self.window_points
                end = start + self.window_points
                window = eeg_data[start:end, 0:64]  # [window_points, channels]
                eeg_samples.append(window)
                ear_labels.append(attention_ear)

        return np.array(eeg_samples), np.array(ear_labels)

    def get_data_loaders(self, subject_indices, test_split=0.2, random_seed=42):
        X_list, y_list = [], []

        for sub_idx in subject_indices:
            X, y = self.load_single_subject(sub_idx)
            print(f"Subject {sub_idx} data shape: X={X.shape}, y={y.shape}")
            X_list.append(X)
            y_list.append(y)

        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)
        dataset = TensorDataset(X, y)

        print(f"\nCombined dataset shape: X={X.shape}, y={y.shape}")
        print(f"Total samples: {len(dataset)}")

        # 计算测试集大小
        test_size = int(len(dataset) * test_split)
        train_size = len(dataset) - test_size

        # 分割数据集
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, test_loader


# ==================== 位置编码模块 ====================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
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
    def __init__(self, d_model, max_len=5000, nhead=4, dropout=0.1,sin=True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead
        self.sin=sin

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
        self.relu=nn.ReLU()

    def forward(self, x):
        res=x
        batch_size, seq_len, _ = x.shape

        if self.sin:
            x=self.sin_encoder(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x
        )

        # 残差连接和归一化
        x = self.norm1(x)
        x = self.relu(x)
        return x+res

class GRUPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=None):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        dense_out = self.dense(gru_out)
        return x + dense_out


class LogicalMaskEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.relu=nn.ReLU()
        self.register_buffer("mask", self.generate_mask(max_len))

    def generate_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()

    def forward(self, x):
        res = x
        batch_size, seq_len, _ = x.shape
        attn_mask = self.mask[:seq_len, :seq_len]
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask
        )
        x = self.norm1(x)
        x = self.relu(x)
        return x + res


class sin_LogicalMskEncoding_mask(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.sin_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        self.self_attn = LogicalMaskEncoding(d_model, max_len)

    def forward(self, x):
        x = self.sin_encoder(x)
        x = self.self_attn(x)
        return x


class sin_Gru(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.sin_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        self.self_gru = GRUPositionalEncoding(d_model)

    def forward(self, x):
        x = self.self_gru(x)
        x = self.sin_encoder(x)
        return x


# ==================== EEG分类模型 ====================
class EEGClassifier(nn.Module):
    def __init__(self, pe_type='sinusoidal', input_channels=64, d_model=64, nhead=4,
                 dim_feedforward=128, num_layers=2, num_classes=2, max_length=128):
        super().__init__()
        # 输入投影层：将EEG通道映射到d_model维
        self.input_proj = nn.Linear(input_channels, d_model)

        # 位置编码选择
        if pe_type == 'sinusoidal':
            self.positional_encoding = Control(d_model,sin=True)
        elif pe_type == 'gru':
            self.positional_encoding = GRUPositionalEncoding(d_model)
        elif pe_type == 'Logical':
            self.positional_encoding = LogicalMaskEncoding(d_model, max_length)
        elif pe_type == 'sin_gru':
            self.positional_encoding = sin_Gru(d_model, max_length)
        elif pe_type == 'sin_Logical':
            self.positional_encoding = sin_LogicalMskEncoding_mask(d_model, max_length)
        elif pe_type == 'none':  # 无位置编码
            self.positional_encoding = Control(d_model,sin=False)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.3
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 分类头
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 输入投影：[batch, seq_len, channels] -> [batch, seq_len, d_model]
        x = self.input_proj(x)

        # 位置编码
        x = self.positional_encoding(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 池化和分类
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


# ==================== 训练和评估函数 ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss /= len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses, train_accs, val_accs


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

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = total_loss / len(test_loader.dataset)

    # 计算分类指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    return test_loss, accuracy, precision, recall, f1, all_preds, all_labels


# ==================== 可视化函数 ====================
def plot_metrics(train_losses, val_losses, train_accs, val_accs, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # plt.savefig(f'{model_name}_metrics.png')
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, model_name):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Right Ear', 'Left Ear']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'{model_name} Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # 在格子中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()


# ==================== 主函数 ====================
def main():
    # 加载EEG数据
    DATA_ROOT = r'EEGdata'  # 修改为你的数据路径
    SUBJECT_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16]  # 使用10个被试的数据
    WINDOW_POINTS = 256  # 时间窗口大小
    CHANNELS = 64  # EEG通道数

    print("Loading EEG dataset...")
    eeg_data = EEGData_KULAAD(
        data_root=DATA_ROOT,
        batch_size=64,
        window_points=WINDOW_POINTS
    )

    # 获取数据加载器
    train_loader, test_loader = eeg_data.get_data_loaders(
        subject_indices=SUBJECT_INDICES,
        test_split=0.2,
        random_seed=seed
    )

    # 使用部分训练数据作为验证集
    train_size = len(train_loader.dataset)
    val_size = int(train_size * 0.2)
    train_size -= val_size

    train_dataset, val_dataset = random_split(
        train_loader.dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 超参数
    d_model = 64  # 减小模型尺寸防止过拟合
    nhead = 8
    num_layers = 3
    epochs = 20
    lr = 0.0005

    # 位置编码类型
    pe_types = ['none', 'sinusoidal', 'gru', 'sin_gru', 'Logical', 'sin_Logical']
    results = {}
    all_metrics = []

    for pe_type in pe_types:
        print(f"\n{'=' * 50}")
        print(f"Training model with {pe_type if pe_type != 'none' else 'no'} positional encoding")
        print(f"{'=' * 50}")

        # 创建模型
        model = EEGClassifier(
            pe_type=pe_type,
            input_channels=CHANNELS,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            num_layers=num_layers,
            max_length=WINDOW_POINTS
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        start_time = time.time()
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion,
            optimizer, num_epochs=epochs
        )
        training_time = time.time() - start_time

        # 评估模型
        test_loss, accuracy, precision, recall, f1, preds, labels = evaluate_model(
            model, test_loader, criterion
        )

        # 存储结果
        results[pe_type] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'preds': preds,
            'labels': labels,
            'training_time': training_time
        }

        # 可视化
        # plot_metrics(train_losses, val_losses, train_accs, val_accs, f"{pe_type}_pe")
        # plot_confusion_matrix(labels, preds, f"{pe_type}_pe")

        # 记录指标
        all_metrics.append({
            'Positional Encoding': pe_type,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Training Time (s)': training_time
        })

    # 结果比较
    print("\nFinal Comparison of All Models:")
    print("{:<20} {:<10} {:<10} {:<10} {:<10} {:<15}".format(
        'Positional Encoding', 'Accuracy', 'Precision', 'Recall', 'F1', 'Training Time (s)'))

    for metrics in all_metrics:
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<15.2f}".format(
            metrics['Positional Encoding'],
            metrics['Accuracy'],
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1 Score'],
            metrics['Training Time (s)']))

    # 绘制所有模型的准确率曲线对比
    plt.figure(figsize=(12, 8))
    for pe_type in pe_types:
        if pe_type in results:
            plt.plot(results[pe_type]['val_accs'], label=f"{pe_type} PE")

    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    # plt.savefig('all_models_val_accuracy.png')
    plt.show()


if __name__ == "__main__":
    main()