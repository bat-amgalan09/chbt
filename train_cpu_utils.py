import torch
import psutil
import time
import threading
from queue import Queue
import threading


## Cpu Training model
def train_model(rank, model, dataloader, criterion, optimizer, epochs=10, world_size=1, queue=None):
    model.train()
    process = psutil.Process()
    times = []
    mem_usage = []
    throughputs = []
    energies = []
    grad_times = []
    accuracies = []
    total_samples = 0
    power_watts = 65 #For MacBook CPU
    # power_watts = 250  for NVIDIA A100 GPU
   

    for epoch in range(epochs):
        start_time = time.time()
        epoch_accuracy = 0
        total_samples_epoch = 0
        grad_time_epoch = 0
        cpu_usages = []
        # Set model to training mode
        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(model.embedding.weight.device), target_seq.to(model.embedding.weight.device)
            optimizer.zero_grad()
            output = model(input_seq, target_seq[:, :-1])
            loss = criterion(output.view(-1, model.fc.out_features), target_seq[:, 1:].contiguous().view(-1))

            # Computing the accuracy
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

            total_samples_epoch += input_seq.size(0)
            cpu_usages.append(psutil.cpu_percent(interval=None))

        epoch_time = time.time() - start_time
        times.append(epoch_time)
        mem_usage.append(process.memory_info().rss / 1024 ** 2)
        throughput = total_samples_epoch / epoch_time
        throughputs.append(throughput)
        avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
        energy = avg_cpu_usage * epoch_time * power_watts / 100
        energies.append(energy)
        grad_times.append(grad_time_epoch)
        accuracies.append(epoch_accuracy / len(dataloader))

        print(f"Rank {rank}, Epoch {epoch+1}, Accuracy: {accuracies[-1]:.2%}, Time: {epoch_time:.2f}s, "
              f"Memory: {mem_usage[-1]:.2f}MB, Throughput: {throughput:.2f} samples/s, "
              f"Energy: {energy:.2f}J, Gradient Time: {grad_time_epoch:.2f}s")

        total_samples += total_samples_epoch

    if queue is not None:
        queue.put((rank, times, mem_usage, throughputs, energies, grad_times, accuracies))

    return times, mem_usage, throughputs, energies, grad_times, accuracies


## Multi-threaded CPU training model
# This function is designed to train a model using multiple threads.
# It splits the dataset into chunks and trains each chunk in a separate thread.
def train_model_threaded(rank, model, dataloader, criterion, optimizer, epochs=10, num_threads=4):
    model.train()
    process = psutil.Process()
    times = []
    mem_usage = []
    throughputs = []
    energies = []
    grad_times = []
    accuracies = []
    total_samples = 0
    power_watts = 65 

    # Split dataloader into chunks for threads
    dataset = dataloader.dataset
    dataset_size = len(dataset)
    chunk_size = dataset_size // num_threads
    subsets = [torch.utils.data.Subset(dataset, range(i * chunk_size, min((i + 1) * chunk_size, dataset_size))) for i in range(num_threads)]
    dataloaders = [torch.utils.data.DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False, num_workers=0) for subset in subsets]
    #Training chunk of data for each thread
    def train_chunk(dataloader, model, criterion, optimizer, result_queue):
        chunk_accuracy = 0
        chunk_samples = 0
        chunk_grad_time = 0
        chunk_cpu_usages = []

        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(model.embedding.weight.device), target_seq.to(model.embedding.weight.device)
            optimizer.zero_grad()
            output = model(input_seq, target_seq[:, :-1])
            loss = criterion(output.view(-1, model.fc.out_features), target_seq[:, 1:].contiguous().view(-1))

            #accuracy
            predicted_tokens = torch.argmax(output, dim=-1)
            target = target_seq[:, 1:].contiguous()
            mask = target != criterion.ignore_index
            correct = (predicted_tokens == target) & mask
            correct_tokens = correct.sum().item()
            total_tokens = mask.sum().item()
            batch_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
            chunk_accuracy += batch_accuracy

            # Backward pass
            grad_start = time.time()
            loss.backward()
            chunk_grad_time += time.time() - grad_start
            optimizer.step()

            chunk_samples += input_seq.size(0)
            chunk_cpu_usages.append(psutil.cpu_percent(interval=None))

        result_queue.put((chunk_accuracy, chunk_samples, chunk_grad_time, chunk_cpu_usages))
    #Training the model for each epoch
    for epoch in range(epochs):
        start_time = time.time()
        result_queue = Queue()
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=train_chunk, args=(dataloaders[i], model, criterion, optimizer, result_queue))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        epoch_accuracy = 0
        total_samples_epoch = 0
        grad_time_epoch = 0
        cpu_usages = []
        for _ in range(num_threads):
            chunk_accuracy, chunk_samples, chunk_grad_time, chunk_cpu_usages = result_queue.get()
            epoch_accuracy += chunk_accuracy
            total_samples_epoch += chunk_samples
            grad_time_epoch += chunk_grad_time
            cpu_usages.extend(chunk_cpu_usages)

        epoch_time = time.time() - start_time
        times.append(epoch_time)
        mem_usage.append(process.memory_info().rss / 1024 ** 2)
        throughput = total_samples_epoch / epoch_time
        throughputs.append(throughput)
        avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 1.0
        energy = avg_cpu_usage * epoch_time * power_watts / 100
        energies.append(energy)
        grad_times.append(grad_time_epoch)
        accuracies.append(epoch_accuracy / sum(len(dl) for dl in dataloaders))

        print(f"Threaded Epoch {epoch+1}, Accuracy: {accuracies[-1]:.2%}, Time: {epoch_time:.2f}s, "
              f"Memory: {mem_usage[-1]:.2f}MB, Throughput: {throughput:.2f} samples/s, "
              f"Energy: {energy:.2f}J, Gradient Time: {grad_time_epoch:.2f}s")

        total_samples += total_samples_epoch

    return times, mem_usage, throughputs, energies, grad_times, accuracies

 

