import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# 固定随机种子
torch.manual_seed(43)
np.random.seed(43)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 数据加载与预处理
def build_vocab(texts, min_freq=5):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word_counts = {}

    # 统计词频
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    # 构建词汇表
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].split()[:self.max_length]
        sequence = [self.vocab.get(word, 1) for word in text]  # 1 for <UNK>
        sequence = sequence + [0] * (self.max_length - len(sequence))  # 0 for <PAD>
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 2. 位置编码模块（保持不变）
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
        self.relu = nn.ReLU()

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
        x = self.relu(x)
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
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(d_model)
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


# 3. 主模型（修改为分类模型）
class TextClassifier(nn.Module):
    def __init__(self, pe_type='sinusoidal', vocab_size=20000, d_model=128, nhead=4,
                 dim_feedforward=256, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

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

        # 分类头
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 嵌入层
        x = self.embedding(x)

        # 位置编码
        x = self.positional_encoding(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 池化和分类
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


# 4. 训练和评估函数（修改为分类任务，添加早停机制）
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=5):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

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
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        # 早停检查
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | '
              f'Best Val Acc: {best_val_acc:.4f} | No Improve: {epochs_no_improve}/{patience}')

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
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


# 5. 可视化函数（调整为分类任务）
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
    # plt.show()


def plot_confusion_matrix(all_labels, all_preds, model_name):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Negative', 'Positive']
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

    # plt.savefig(f'{model_name}_confusion_matrix.png')
    # plt.show()


# 主函数
def main():
    # 加载IMDB数据集
    print("Loading IMDB dataset...")
    train_data = pd.read_csv("archive/train.csv")
    test_data = pd.read_csv("archive/test.csv")

    # 标签映射
    label_mapping = {'neg': 0, 'pos': 1}
    train_data['label'] = train_data['label'].map(label_mapping)
    test_data['label'] = test_data['label'].map(label_mapping)

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['text'].values,
        train_data['label'].values,
        test_size=0.2,
        random_state=42
    )

    test_texts = test_data['text'].values
    test_labels = test_data['label'].values

    # 构建词汇表
    print("Building vocabulary...")
    vocab = build_vocab(train_texts)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # 超参数
    max_length = 200
    batch_size = 64
    epochs = 10
    lr = 0.0005
    d_model = 128
    nhead = 8
    dim_feedforward = 256
    num_layers = 3
    patience = 3  # 早停耐心值

    # 创建数据集和数据加载器
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, vocab, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 位置编码类型
    pe_types = [
        'none',
        'sinusoidal',
        'gru',
        'sin_gru',
        'Logical',
        'sin_Logical']
    results = {}
    all_metrics = []

    for pe_type in pe_types:
        print(f"\n{'=' * 50}")
        print(f"Training model with {pe_type if pe_type != 'none' else 'no'} positional encoding")
        print(f"{'=' * 50}")

        # 创建模型
        model = TextClassifier(
            pe_type=pe_type,
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers
        ).to(device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        start_time = time.time()
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion,
            optimizer, num_epochs=epochs, patience=patience
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