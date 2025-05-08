import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer
import numpy as np
from data_utils import prepare_dailydialog_data
import psutil

# Dataset
class ChatDataset(Dataset):
    def __init__(self, input_file, target_file, max_len=30):
        self.input_tokens = np.load(input_file)
        self.target_tokens = np.load(target_file)
        self.max_len = max_len

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.input_tokens[idx]), torch.tensor(self.target_tokens[idx])

# Model
class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.3):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, target_seq=None, max_len=30, temperature=0.7):
        embedded = self.embedding(input_seq)
        encoder_output, (hidden, cell) = self.encoder(embedded)
        if target_seq is not None:
            embedded_target = self.embedding(target_seq)
            decoder_output, _ = self.decoder(embedded_target, (hidden, cell))
            output = self.fc(decoder_output)
            return output
        else:
            return None  # for simplicity

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def train(rank, world_size):
    setup_ddp(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    prepare_dailydialog_data(tokenizer, limit=1000)

    train_data = ChatDataset("train_input_tokens.npy", "train_target_tokens.npy")
    sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = ChatbotModel(tokenizer.vocab_size).to(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    process = psutil.Process()
    power_watts = 250  # GPU estimate

    times, mem_usage, throughputs, energies, grad_times, accuracies = [], [], [], [], [], []

    for epoch in range(3):
        ddp_model.train()
        start_time = time.time()
        grad_time_epoch = 0
        total_loss = 0
        correct = 0
        total_tokens = 0

        for input_seq, target_seq in train_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            optimizer.zero_grad()
            grad_start = time.time()
            output = ddp_model(input_seq, target_seq[:, :-1])
            grad_time_epoch += time.time() - grad_start
            loss = criterion(output.view(-1, output.shape[-1]), target_seq[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred = output.argmax(dim=-1)
            mask = target_seq[:, 1:] != tokenizer.pad_token_id
            correct += ((pred == target_seq[:, 1:]) * mask).sum().item()
            total_tokens += mask.sum().item()

        epoch_time = time.time() - start_time
        accuracy = correct / total_tokens if total_tokens > 0 else 0
        memory = process.memory_info().rss / (1024 * 1024)
        throughput = len(train_loader.dataset) / epoch_time
        energy = power_watts * epoch_time

        times.append(epoch_time)
        mem_usage.append(memory)
        throughputs.append(throughput)
        energies.append(energy)
        grad_times.append(grad_time_epoch)
        accuracies.append(accuracy)

        print(f"Rank {rank}, Epoch {epoch+1}: Time={epoch_time:.2f}s, Loss={total_loss/len(train_loader):.4f}, "
              f"Mem={memory:.2f}MB, Throughput={throughput:.2f} samples/s, "
              f"Energy={energy:.2f}J, Grad Time={grad_time_epoch:.2f}s, Acc={accuracy:.2%}")

    cleanup_ddp()

if __name__ == "__main__":
    world_size = 1
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
