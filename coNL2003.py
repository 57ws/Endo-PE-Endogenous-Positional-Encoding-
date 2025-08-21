import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score, precision_score, recall_score
import time
import os
import requests
import zipfile
from collections import defaultdict
from transformers import AutoTokenizer

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 自动下载CoNLL-2003数据集
def download_conll2003():
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)

    # 检查是否已下载
    if os.path.exists(os.path.join(base_dir, "eng.train")) and \
            os.path.exists(os.path.join(base_dir, "eng.testa")) and \
            os.path.exists(os.path.join(base_dir, "eng.testb")):
        print("CoNLL-2003 dataset already exists.")
        return base_dir

    print("Downloading CoNLL-2003 dataset...")
    url = "https://data.deepai.org/conll2003.zip"
    response = requests.get(url, stream=True)
    zip_path = os.path.join(base_dir, "conll2003.zip")

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    # 解压文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)

    # 删除zip文件
    os.remove(zip_path)
    print("CoNLL-2003 dataset downloaded and extracted.")
    return base_dir


# 解析CoNLL-2003格式的文件
def load_conll2003_file(file_path):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                parts = line.split()
                # CoNLL-2003格式: word POS TAG NER
                word = parts[0]
                ner_tag = parts[-1]  # 最后一列是NER标签
                current_sentence.append(word)
                current_labels.append(ner_tag)

    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)

    return sentences, labels


# 加载CoNLL-2003数据集
def load_conll2003_data(data_dir):
    train_file = os.path.join(data_dir, "train.txt")
    dev_file = os.path.join(data_dir, "valid.txt")
    test_file = os.path.join(data_dir, "test.txt")

    train_sentences, train_labels = load_conll2003_file(train_file)
    dev_sentences, dev_labels = load_conll2003_file(dev_file)
    test_sentences, test_labels = load_conll2003_file(test_file)

    return (train_sentences, train_labels), (dev_sentences, dev_labels), (test_sentences, test_labels)


# 构建词汇表和标签映射
def build_vocab_and_tag_map(sentences, labels, min_freq=2):
    word_counts = defaultdict(int)
    tag_set = set()

    for sent, tags in zip(sentences, labels):
        for word in sent:
            word_counts[word] += 1
        for tag in tags:
            tag_set.add(tag)

    # 构建词汇表
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    # 构建标签映射：保留0用于填充，其他标签从1开始
    tag_map = {'<PAD>': 0}  # 明确将0索引分配给填充标签
    for tag in sorted(tag_set):
        tag_map[tag] = len(tag_map)  # 其他标签从1开始索引

    return vocab, tag_map

# 数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, labels, vocab, tag_map, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.tag_map = tag_map
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx][:self.max_length]
        tags = self.labels[idx][:self.max_length]

        # 转换为索引
        word_indices = [self.vocab.get(word, 1) for word in sentence]  # 1 for <UNK>
        tag_indices = [self.tag_map[tag] for tag in tags]

        # 填充
        word_indices = word_indices + [0] * (self.max_length - len(word_indices))  # 0 for <PAD>
        tag_indices = tag_indices + [0] * (self.max_length - len(tag_indices))  # 0 for <PAD> (假设0是PAD标签)

        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)


# 位置编码模块
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


# NER模型
class NERModel(nn.Module):
    def __init__(self, pe_type='sinusoidal', vocab_size=20000, d_model=128, nhead=4,
                 dim_feedforward=256, num_layers=2, num_tags=9):
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

        # 分类头 - 每个位置一个分类
        self.fc = nn.Linear(d_model, num_tags)

    def forward(self, x):
        # 嵌入层
        x = self.embedding(x)

        # 位置编码
        x = self.positional_encoding(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 分类
        x = self.fc(x)
        return x


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 忽略填充部分
            mask = (inputs != 0)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算损失，忽略填充部分
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
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

                # 计算损失，忽略填充部分
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                epoch_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_ner_model.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_ner_model.pth'))
    return train_losses, val_losses


# 评估函数
def evaluate_model(model, test_loader, tag_map):
    model.eval()
    all_preds = []
    all_labels = []

    # 反转标签映射
    idx_to_tag = {idx: tag for tag, idx in tag_map.items()}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)

            # 转换为CPU numpy数组
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            inputs = inputs.cpu().numpy()

            # 处理每个句子
            for i in range(len(inputs)):
                # 找到非填充部分
                non_pad_mask = inputs[i] != 0
                sentence_preds = preds[i][non_pad_mask]
                sentence_labels = labels[i][non_pad_mask]

                # 转换为标签名称
                pred_tags = [idx_to_tag.get(idx, 'O') for idx in sentence_preds]
                true_tags = [idx_to_tag.get(idx, 'O') for idx in sentence_labels]

                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    # 使用seqeval计算指标
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print("\nDetailed Classification Report:")
    print(seqeval_classification_report(all_labels, all_preds))

    return precision, recall, f1, all_preds, all_labels


# 可视化函数
def plot_metrics(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    # plt.savefig(f'{model_name}_loss.png')
    # plt.show()


# 主函数
def main():
    # 下载并加载数据
    data_dir = download_conll2003()
    (train_sentences, train_labels), (dev_sentences, dev_labels), (test_sentences, test_labels) = load_conll2003_data(
        data_dir)

    # 构建词汇表和标签映射
    vocab, tag_map = build_vocab_and_tag_map(train_sentences, train_labels)
    vocab_size = len(vocab)
    num_tags = len(tag_map)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of tags: {num_tags}")
    print(f"Tags: {list(tag_map.keys())}")

    # 超参数
    max_length = 128
    batch_size = 32
    epochs = 10
    lr = 0.001
    d_model = 128
    nhead = 4
    dim_feedforward = 256
    num_layers = 2

    # 创建数据集和数据加载器
    train_dataset = NERDataset(train_sentences, train_labels, vocab, tag_map, max_length)
    dev_dataset = NERDataset(dev_sentences, dev_labels, vocab, tag_map, max_length)
    test_dataset = NERDataset(test_sentences, test_labels, vocab, tag_map, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 位置编码类型
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
        model = NERModel(
            pe_type=pe_type,
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            num_tags=num_tags
        ).to(device)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标签
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        start_time = time.time()
        train_losses, val_losses = train_model(
            model, train_loader, dev_loader, criterion,
            optimizer, num_epochs=epochs
        )
        training_time = time.time() - start_time

        # 评估模型
        precision, recall, f1, preds, labels = evaluate_model(model, test_loader, tag_map)

        # 存储结果
        results[pe_type] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'preds': preds,
            'labels': labels,
            'training_time': training_time
        }

        # 可视化
        # plot_metrics(train_losses, val_losses, f"{pe_type}_pe")

        # 记录指标
        all_metrics.append({
            'Positional Encoding': pe_type,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Training Time (s)': training_time
        })

    # 结果比较
    print("\nFinal Comparison of All Models:")
    print("{:<20} {:<10} {:<10} {:<10} {:<15}".format(
        'Positional Encoding', 'Precision', 'Recall', 'F1', 'Training Time (s)'))

    for metrics in all_metrics:
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<15.2f}".format(
            metrics['Positional Encoding'],
            metrics['Precision'],
            metrics['Recall'],
            metrics['F1 Score'],
            metrics['Training Time (s)']))

    # 绘制所有模型的F1分数对比
    plt.figure(figsize=(12, 6))
    f1_scores = [metrics['F1 Score'] for metrics in all_metrics]
    pe_names = [metrics['Positional Encoding'] for metrics in all_metrics]

    plt.bar(pe_names, f1_scores)
    plt.xlabel('Positional Encoding Type')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison of Different Positional Encodings')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    # plt.savefig('all_models_f1_comparison.png')
    plt.show()


if __name__ == "__main__":
    main()