import os
import json
import time
import torch
import psutil
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import deepspeed


BATCH_SIZE = 64
EPOCHS = 5
MAX_LEN = 30
LIMIT = 5000
DS_CONFIG_FILE = "ds_config.json"


def prepare_dailydialog_data(tokenizer, max_len=30, limit=5000):
    dataset = load_dataset('daily_dialog', split='train')
    dialog_pairs = []
    for dialog in dataset['dialog']:
        for i in range(len(dialog) - 1):
            input_text = dialog[i][:500]
            target_text = dialog[i + 1][:500]
            if input_text.strip() and target_text.strip():
                dialog_pairs.append((input_text, target_text))
            if len(dialog_pairs) >= limit:
                break
        if len(dialog_pairs) >= limit:
            break

    input_tokens_list = []
    target_tokens_list = []
    for input_text, target_text in dialog_pairs:
        input_tokens = tokenizer.encode(input_text, max_length=max_len, truncation=True, padding='max_length')
        target_tokens = tokenizer.encode(target_text, max_length=max_len, truncation=True, padding='max_length')
        input_tokens_list.append(input_tokens)
        target_tokens_list.append(target_tokens)

    np.save('train_input_tokens.npy', np.array(input_tokens_list))
    np.save('train_target_tokens.npy', np.array(target_tokens_list))


class ChatDataset(Dataset):
    def __init__(self, input_file, target_file):
        self.input_tokens = np.load(input_file)
        self.target_tokens = np.load(target_file)

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.input_tokens[idx]), torch.tensor(self.target_tokens[idx])


class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, target_seq):
        embedded = self.embedding(input_seq)
        _, (hidden, cell) = self.encoder(embedded)
        embedded_target = self.embedding(target_seq)
        decoder_output, _ = self.decoder(embedded_target, (hidden, cell))
        output = self.fc(decoder_output)
        return output


ds_config = {
    "train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "fp16": {
        "enabled": True
    }
}

with open(DS_CONFIG_FILE, "w") as f:
    json.dump(ds_config, f, indent=2)


def train_deepspeed():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    prepare_dailydialog_data(tokenizer, max_len=MAX_LEN, limit=LIMIT)

    train_data = ChatDataset("train_input_tokens.npy", "train_target_tokens.npy")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    model = ChatbotModel(tokenizer.vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=DS_CONFIG_FILE
    )

    process = psutil.Process()

    for epoch in range(EPOCHS):
        model_engine.train()
        start = time.time()
        total_loss, correct, total_tokens, grad_time = 0, 0, 0, 0

        for input_seq, target_seq in train_loader:
            input_seq, target_seq = input_seq.to(model_engine.device), target_seq.to(model_engine.device)
            t0 = time.time()
            output = model_engine(input_seq, target_seq[:, :-1])
            grad_time += time.time() - t0
            loss = criterion(output.view(-1, output.shape[-1]), target_seq[:, 1:].reshape(-1))
            model_engine.backward(loss)
            model_engine.step()
            total_loss += loss.item()

            pred = output.argmax(dim=-1)
            mask = target_seq[:, 1:] != tokenizer.pad_token_id
            correct += ((pred == target_seq[:, 1:]) * mask).sum().item()
            total_tokens += mask.sum().item()

        acc = correct / total_tokens if total_tokens else 0
        elapsed = time.time() - start
        mem = process.memory_info().rss / 1024**2
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Time={elapsed:.2f}s, GradTime={grad_time:.2f}s, Mem={mem:.2f}MB, Acc={acc:.2%}")

if __name__ == "__main__":
    train_deepspeed()
