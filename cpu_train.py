import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from multiprocessing import Queue
import psutil
from nltk.translate.bleu_score import corpus_bleu
from train_cpu import train_model, train_model_threaded


# Set environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set device to CPU
device = torch.device('cpu')

# Preprocess dataset
dataset = load_dataset('daily_dialog', split='train')
dialog_pairs = []
for dialog in dataset['dialog']:
    for i in range(len(dialog) - 1):
        input_text = dialog[i][:500]
        target_text = dialog[i + 1][:500]
        if input_text.strip() and target_text.strip():
            dialog_pairs.append((input_text, target_text))
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Fix padding token issue
vocab_size = tokenizer.vocab_size
input_tokens_list = []
target_tokens_list = []
# Tokenize and encode the dataset Maximum length of 30 tokens to make it not too long
max_len = 30
#Encode the input and target texts
for input_text, target_text in dialog_pairs:
    input_tokens = tokenizer.encode(input_text, max_length=max_len, truncation=True, padding='max_length')
    target_tokens = tokenizer.encode(target_text, max_length=max_len, truncation=True, padding='max_length')
    input_tokens_list.append(input_tokens)
    target_tokens_list.append(target_tokens)

all_data = list(zip(input_tokens_list, target_tokens_list))
np.random.shuffle(all_data)
train_data = all_data[:int(0.8 * len(all_data))]
test_data = all_data[int(0.8 * len(all_data)):]


np.save('train_input_tokens.npy', np.array([d[0] for d in train_data]))
np.save('train_target_tokens.npy', np.array([d[1] for d in train_data]))
np.save('test_input_tokens.npy', np.array([d[0] for d in test_data]))
np.save('test_target_tokens.npy', np.array([d[1] for d in test_data]))

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

#Chatbot Model
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
    # Forward pass 
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


# Chatting with the bot
def chat_with_bot(model, tokenizer, max_len=50, temperature=0.7):
    model.eval()
    model.to(device)
    latencies = []
    print("Ask me Something(type quit to stop)...")
    while True:
        input_text = input("You: ")
        if input_text.lower() in ['quit']:
            break
        start_time = time.time()
        try:
            input_tokens = tokenizer.encode(input_text, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt').to(device)
            with torch.no_grad():
                output = model(input_tokens, max_len=max_len, temperature=temperature)
                predicted_ids = output
                response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        latency = time.time() - start_time
        latencies.append(latency)
        print(f"Bot: {response}, Latency: {latency:.4f}s")
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average Inference Latency: {avg_latency:.4f}s")
    return avg_latency

if __name__ == '__main__':
    train_data = ChatDataset('train_input_tokens.npy', 'train_target_tokens.npy')
    test_data = ChatDataset('test_input_tokens.npy', 'test_target_tokens.npy')
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    model = ChatbotModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting single-core training...")
    times_single, mem_single, throughputs_single, energies_single, grad_times_single, accuracies_single = train_model(
        0, model, train_dataloader, criterion, optimizer, epochs=10
    )

    total_time_single = sum(times_single)
    total_mem_single = sum(mem_single)
    avg_accuracy_single = sum(accuracies_single) / len(accuracies_single)
    print(f"\nSingle-core Totals:")
    print(f"Total Time: {total_time_single:.2f} seconds")
    print(f"Total Memory Usage: {total_mem_single:.2f} MB")
    print(f"Average Accuracy: {avg_accuracy_single:.2%}")
 
    cpu_sizes = [4]
    all_times = [times_single]
    all_mem = [mem_single]
    all_throughputs = [throughputs_single]
    all_energies = [energies_single]
    all_grad_times = [grad_times_single]
    all_accuracies = [accuracies_single]

    for cpu_sizes in cpu_sizes:
        print(f"\nStarting {cpu_sizes}-core training...")
        queue = Queue()
        processes = []
        mp.set_start_method('spawn', force=True)

        dataset_size = len(train_data)
        chunk_size = dataset_size // cpu_sizes
        subsets = [Subset(train_data, range(i * chunk_size, (i + 1) * chunk_size)) for i in range(cpu_sizes)]
        if dataset_size % cpu_sizes != 0:
            subsets[-1] = Subset(train_data, range((cpu_sizes - 1) * chunk_size, dataset_size))

        for rank in range(cpu_sizes):
            model_copy = ChatbotModel(vocab_size).to(device)
            model_copy.load_state_dict(model.state_dict())
            optimizer_copy = optim.Adam(model_copy.parameters(), lr=0.001)
            sub_dataloader = DataLoader(subsets[rank], batch_size=64, shuffle=True, num_workers=0)
            p = mp.Process(target=train_model, args=(rank, model_copy, sub_dataloader, criterion, optimizer_copy, 10, cpu_sizes, queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        process_metrics = []
        for _ in range(cpu_sizes):
            rank, times, mem_usage, throughputs, energies, grad_times, accuracies = queue.get()
            process_metrics.append((rank, times, mem_usage, throughputs, energies, grad_times, accuracies))

        process_metrics.sort(key=lambda x: x[0])

        print(f"\nIndividual Process Metrics for {cpu_sizes}-core:")
        for rank, times, mem_usage, throughputs, energies, grad_times, accuracies in process_metrics:
            total_time_proc = sum(times)
            total_mem_proc = sum(mem_usage)
            avg_accuracy_proc = sum(accuracies) / len(accuracies)
            print(f"Process {rank}: Total Time: {total_time_proc:.2f}s, Total Memory: {total_mem_proc:.2f}MB, Avg Accuracy: {avg_accuracy_proc:.2%}")

        times_multi = [max([proc[1][i] for proc in process_metrics]) for i in range(len(process_metrics[0][1]))]
        mem_multi = [sum([proc[2][i] for proc in process_metrics]) for i in range(len(process_metrics[0][2]))]
        throughputs_multi = [sum([proc[3][i] for proc in process_metrics]) for i in range(len(process_metrics[0][3]))]
        energies_multi = [sum([proc[4][i] for proc in process_metrics]) for i in range(len(process_metrics[0][4]))]
        grad_times_multi = [sum([proc[5][i] for proc in process_metrics]) for i in range(len(process_metrics[0][5]))]
        accuracies_multi = [sum([proc[6][i] for proc in process_metrics]) / cpu_sizes for i in range(len(process_metrics[0][6]))]

        total_time_multi = sum(times_multi)
        total_mem_multi = sum(mem_multi)
        avg_accuracy_multi = sum(accuracies_multi) / len(accuracies_multi)
        print(f"\n{cpu_sizes}-core Totals:")
        print(f"Total Time: {total_time_multi:.2f} seconds")
        print(f"Total Memory Usage: {total_mem_multi:.2f} MB")
        print(f"Average Accuracy: {avg_accuracy_multi:.2%}")

        all_times.append(times_multi)
        all_mem.append(mem_multi)
        all_throughputs.append(throughputs_multi)
        all_energies.append(energies_multi)
        all_grad_times.append(grad_times_multi)
        all_accuracies.append(accuracies_multi)

    thread_counts = [4]
    all_times_threaded = []
    all_mem_threaded = []
    all_throughputs_threaded = []
    all_energies_threaded = []
    all_grad_times_threaded = []
    all_accuracies_threaded = []

    for num_threads in thread_counts:
        print(f"\nStarting {num_threads}-thread training...")
        model_threaded = ChatbotModel(vocab_size).to(device)
        model_threaded.load_state_dict(model.state_dict())
        optimizer_threaded = optim.Adam(model_threaded.parameters(), lr=0.001)
        times, mem, throughputs, energies, grad_times, accuracies = train_model_threaded(
            0, model_threaded, train_dataloader, criterion, optimizer_threaded, epochs=10, num_threads=num_threads
        )
        total_time_threaded = sum(times)
        total_mem_threaded = sum(mem)
        avg_accuracy_threaded = sum(accuracies) / len(accuracies)
        print(f"\n{num_threads}-thread Totals:")
        print(f"Total Time: {total_time_threaded:.2f} seconds")
        print(f"Total Memory Usage: {total_mem_threaded:.2f} MB")
        print(f"Average Accuracy: {avg_accuracy_threaded:.2%}")
        all_times_threaded.append(times)
        all_mem_threaded.append(mem)
        all_throughputs_threaded.append(throughputs)
        all_energies_threaded.append(energies)
        all_grad_times_threaded.append(grad_times)
        all_accuracies_threaded.append(accuracies)

    epochs = range(1, 11)
    labels = ['Single-core', '4-core', '4-thread']
    all_times.extend(all_times_threaded)
    all_mem.extend(all_mem_threaded)
    all_throughputs.extend(all_throughputs_threaded)
    all_energies.extend(all_energies_threaded)
    all_grad_times.extend(all_grad_times_threaded)
    all_accuracies.extend(all_accuracies_threaded)

    plt.figure(figsize=(15, 12))
    plt.subplot(2, 3, 1)
    for i, (times, label) in enumerate(zip(all_times, labels)):
        plt.plot(epochs, times, label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time per Epoch (s)')
    plt.title('Training Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    for i, (mem, label) in enumerate(zip(all_mem, labels)):
        plt.plot(epochs, mem, label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    for i, (throughput, label) in enumerate(zip(all_throughputs, labels)):
        plt.plot(epochs, throughput, label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Throughput (samples/s)')
    plt.title('Throughput')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    for i, (energy, label) in enumerate(zip(all_energies, labels)):
        plt.plot(epochs, energy, label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Energy (CPU usage * time)')
    plt.title('Energy Consumption')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    for i, (grad_time, label) in enumerate(zip(all_grad_times, labels)):
        plt.plot(epochs, grad_time, label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Time (s)')
    plt.title('Gradient Computation Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 6)
    for i, (accuracy, label) in enumerate(zip(all_accuracies, labels)):
        plt.plot(epochs, [acc * 100 for acc in accuracy], label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('cpu_metrics_with_accuracy.png')

    avg_times_single = sum(times_single) / len(times_single)
    avg_times_4core = sum(all_times[1]) / len(all_times[1])
    avg_times_4thread = sum(all_times_threaded[0]) / len(all_times_threaded[0])

    speedup_4core = avg_times_single / avg_times_4core
    speedup_4thread = avg_times_single / avg_times_4thread
    efficiency_4core = speedup_4core / 4
    efficiency_4thread = speedup_4thread / 4

    print(f"Speedup (4-core): {speedup_4core:.2f}x, Efficiency: {efficiency_4core:.2f}")
    print(f"Speedup (4-thread): {speedup_4thread:.2f}x, Efficiency: {efficiency_4thread:.2f}")

    avg_latency = chat_with_bot(model, tokenizer)

   # Preprocess dataset
