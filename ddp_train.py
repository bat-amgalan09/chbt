import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os
import time
import psutil

# Dataset
class ChatDataset(Dataset):
    def __init__(self, input_file, target_file, max_len=50):
        self.input_tokens = np.load(input_file)
        self.target_tokens = np.load(target_file)
        self.max_len = max_len

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.input_tokens[idx]), torch.tensor(self.target_tokens[idx])

# Chatbot Model
class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=3, dropout=0.3):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, input_seq, target_seq=None, max_len=50, temperature=0.7):
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        encoder_output, (hidden, cell) = self.encoder(embedded)
        if target_seq is not None:
            embedded_target = self.embedding(target_seq)
            embedded_target = self.dropout(embedded_target)
            decoder_output, _ = self.decoder(embedded_target, (hidden, cell))
            output = self.fc(decoder_output)
            return output
        else:
            batch_size = input_seq.size(0)
            start_token = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=input_seq.device)
            decoder_input = start_token
            decoder_hidden = (hidden, cell)
            outputs = []
            for _ in range(max_len):
                embedded_input = self.embedding(decoder_input)
                embedded_input = self.dropout(embedded_input)
                decoder_output, decoder_hidden = self.decoder(embedded_input, decoder_hidden)
                output = self.fc(decoder_output)
                probs = torch.softmax(output / temperature, dim=-1)
                predicted_token = torch.multinomial(probs.squeeze(1), num_samples=1)
                outputs.append(predicted_token)
                decoder_input = predicted_token
            output_seq = torch.cat(outputs, dim=1)
            return output_seq

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Single-node DDP
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size, train_dataloader, test_dataloader, tokenizer, epochs=10):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    model = ChatbotModel(tokenizer.vocab_size).to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    process = psutil.Process()
    times = []
    mem_usage = []
    throughputs = []
    energies = []
    grad_times = []
    accuracies = []
    power_watts = 250  # NVIDIA A100 approximate power consumption
    total_samples = 0

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0
        epoch_accuracy = 0
        total_samples_epoch = 0
        cpu_usages = []
        grad_time_epoch = 0

        for input_seq, target_seq in train_dataloader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            optimizer.zero_grad()
            output = model(input_seq, target_seq[:, :-1])
            loss = criterion(output.view(-1, model.module.fc.out_features), target_seq[:, 1:].contiguous().view(-1))
            
            # Compute accuracy
            predicted_tokens = torch.argmax(output, dim=-1)
            target = target_seq[:, 1:].contiguous()
            mask = target != criterion.ignore_index
            correct = (predicted_tokens == target) & mask
            correct_tokens = correct.sum().item()
            total_tokens = mask.sum().item()
            batch_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
            epoch_accuracy += batch_accuracy

            # Backward pass
            grad_start = time.time()
            loss.backward()
            grad_time_epoch += time.time() - grad_start
            optimizer.step()

            epoch_loss += loss.item()
            total_samples_epoch += input_seq.size(0)
            cpu_usages.append(psutil.cpu_percent(interval=None))

        end_time = time.time()
        epoch_time = end_time - start_time
        times.append(epoch_time)
        mem_usage.append(process.memory_info().rss / 1024 ** 2)
        throughput = total_samples_epoch / epoch_time
        throughputs.append(throughput)
        avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
        energy = avg_cpu_usage * epoch_time * power_watts / 10000
        energies.append(energy)
        grad_times.append(grad_time_epoch)
        accuracies.append(epoch_accuracy / len(train_dataloader))

        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader):.4f}, "
                  f"Accuracy: {accuracies[-1]:.2%}, Time: {epoch_time:.2f}s, "
                  f"Memory: {mem_usage[-1]:.2f}MB, Throughput: {throughput:.2f} samples/s, "
                  f"Energy: {energy:.2f}, Gradient Time: {grad_time_epoch:.2f}s")

        total_samples += total_samples_epoch

    if rank == 0:
        # Plotting
        epochs = range(1, 11)
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 3, 1)
        plt.plot(epochs, times, label='DDP', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Time per Epoch (s)')
        plt.title('Training Time')
        plt.legend()
        plt.grid(True)
        # ... similar for other subplots (memory, throughput, energy, grad_times, accuracies)
        plt.tight_layout()
        plt.savefig('./ddp_metrics.png')
        plt.close()

        # Print totals
        total_time = sum(times)
        total_mem = sum(mem_usage)
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"\nDDP Totals:")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Total Memory Usage: {total_mem:.2f} MB")
        print(f"Average Accuracy: {avg_accuracy:.2%}")

    cleanup()
    return model, times, mem_usage, throughputs, energies, grad_times, accuracies

def chat_with_bot(model, tokenizer, max_len=50, temperature=0.7):
    model.eval()
    model = model.module.to(torch.device('cuda:0'))
    latencies = []
    print("Starting conversation (type 'exit' or 'quit' to stop)...")
    while True:
        input_text = input("You: ")
        if input_text.lower() in ['exit', 'quit']:
            break
        start_time = time.time()
        try:
            input_tokens = tokenizer.encode(input_text, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt').to(torch.device('cuda:0'))
            with torch.no_grad():
                output = model(input_tokens, max_len=max_len, temperature=temperature)
                response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        latency = time.time() - start_time
        latencies.append(latency)
        print(f"Bot: {response}, Latency: {latency:.4f}s")
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average Inference Latency: {avg_latency:.4f}s")
    return avg_latency

def main():
    world_size = int(os.environ.get('WORLD_SIZE', '2'))  # Set by SLURM
    train_data = ChatDataset('./train_input_tokens.npy',
                             './train_target_tokens.npy')
    test_data = ChatDataset('./test_input_tokens.npy',
                            './test_target_tokens.npy')
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Spawn processes
    model, times, mem_usage, throughputs, energies, grad_times, accuracies = mp.spawn(
        train_model,
        args=(world_size, train_dataloader, test_dataloader, tokenizer),
        nprocs=world_size,
        join=True
    )[0]  # Get model from rank 0
    
    if dist.get_rank() == 0:
        avg_latency = chat_with_bot(model, tokenizer)

if __name__ == '__main__':
    main()