import json
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import jieba   

# ---------- 模型结构（必须与训练时完全相同）----------
class LSTMPoolModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super(LSTMPoolModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(emb)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool = torch.max(lstm_out, dim=1)[0]
        combined = torch.cat([avg_pool, max_pool], dim=1)
        logits = self.fc(combined)
        return nn.functional.log_softmax(logits, dim=-1)

# ---------- 推理接口类 ----------
class EmotionInference:
    def __init__(self, model_path, vocab_path, config_path, device=None):
        """
        参数:
            model_path: 训练好的模型权重文件 (.pth)
            vocab_path: 词汇表JSON文件，格式 {"word": idx}
            config_path: 模型配置JSON文件，包含 embed_dim, hidden_dim, n_layers, dropout, output_dim, max_len 等
            device: 运行设备（可选），None则自动选择cuda/cpu
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.pad_idx = self.vocab.get('<PAD>', 0)

        # 初始化模型
        self.model = LSTMPoolModel(
            vocab_size=len(self.vocab),
            embed_dim=self.config['embed_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim'],
            n_layers=self.config['n_layers'],
            dropout=self.config['dropout'],
            pad_idx=self.pad_idx
        )
        # 加载权重
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)

        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # 分词器（与训练时保持一致，这里使用jieba；若训练时用空格分词请修改）
        self.tokenizer = lambda x: list(jieba.cut(x.strip()))
        # 最大序列长度（训练时截断的长度，默认128）
        self.max_len = self.config.get('max_len', 128)
        # 情感标签（可选，仅用于展示）
        self.emotion_labels = self.config.get('emotion_labels',
                                              ['emotion1', 'emotion2', 'emotion3', 'emotion4', 'emotion5', 'emotion6'])

    def _text_to_sequence(self, text):
        """文本转索引序列，并截断到max_len"""
        words = self.tokenizer(text)
        seq = [self.vocab.get(word, self.vocab.get('<UNK>', len(self.vocab)-1)) for word in words]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        return seq

    def predict(self, text):
        """预测单条文本，返回6维概率分布 (numpy数组)"""
        seq = self._text_to_sequence(text)
        tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.model(tensor)
            probs = torch.exp(log_probs).cpu().numpy()[0]
        return probs

    def predict_batch(self, texts, batch_size=32):
        """批量预测多条文本，返回形状 (len(texts), 6) 的numpy数组"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            sequences = [self._text_to_sequence(t) for t in batch_texts]
            padded = pad_sequence(
                [torch.tensor(seq, dtype=torch.long) for seq in sequences],
                batch_first=True, padding_value=self.pad_idx
            ).to(self.device)
            with torch.no_grad():
                log_probs = self.model(padded)
                probs = torch.exp(log_probs).cpu().numpy()
                results.append(probs)
        return np.vstack(results)

    def predict_proba(self, text):
        """predict的别名，保持sklearn风格"""
        return self.predict(text)
