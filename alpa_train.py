import alpa
import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.lax import scan
import numpy as np
from transformers import AutoTokenizer
import time
import psutil
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import uuid

alpa.init(cluster="ray")

# Dataset
class RedditChatDataset:
    def __init__(self, input_file, target_file, max_len=50):
        self.input_tokens = np.load(input_file)
        self.target_tokens = np.load(target_file)
        self.max_len = max_len

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return jnp.array(self.input_tokens[idx]), jnp.array(self.target_tokens[idx])

# Model
class ChatbotModelJAX:
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=3, dropout=0.3, key=random.PRNGKey(0)):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        keys = random.split(key, 5 + 2 * num_layers)
        # Xavier initialization
        glorot = lambda shape: random.normal(keys[0], shape) * jnp.sqrt(2.0 / (shape[0] + shape[1]))
        self.embedding = glorot((vocab_size, embed_size))
        self.encoder_lstm_weights = [
            (glorot((hidden_size * 4, embed_size + hidden_size)), jnp.zeros((hidden_size * 4,)))
            for _ in range(num_layers)
        ]
        self.decoder_lstm_weights = [
            (glorot((hidden_size * 4, embed_size + hidden_size)), jnp.zeros((hidden_size * 4,)))
            for _ in range(num_layers)
        ]
        self.fc_weights = glorot((hidden_size, vocab_size))
        self.fc_bias = jnp.zeros((vocab_size,))

    def lstm_cell(self, carry, x, weights, training=True):
        h, c = carry
        combined = jnp.concatenate([x, h], axis=-1)
        gates = jnp.dot(combined, weights[0]) + weights[1]
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * jnp.tanh(c_next)
        # Apply dropout during training
        if training:
            key = random.PRNGKey(int(time.time() * 1000))
            dropout_mask = random.bernoulli(key, 1.0 - self.dropout, h_next.shape)
            h_next = h_next * dropout_mask / (1.0 - self.dropout)
        return (h_next, c_next), h_next

    def encoder(self, input_seq, params, training=True):
        embedded = input_seq @ params.embedding
        carry = [(jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,))) for _ in range(self.num_layers)]
        for layer in range(self.num_layers):
            def step(carry, x):
                return self.lstm_cell(carry, x, params.encoder_lstm_weights[layer], training)
            carry[layer], _ = scan(step, carry[layer], embedded)
        return carry[-1]  # Return final layer's (h, c)

    def decoder(self, target_seq, h, c, params, temperature=0.7, inference=False, training=True):
        if not inference:
            embedded = target_seq @ params.embedding
            carry = [(h, c) for _ in range(self.num_layers)]
            outputs = []
            for layer in range(self.num_layers):
                def step(carry, x):
                    return self.lstm_cell(carry, x, params.decoder_lstm_weights[layer], training)
                carry[layer], layer_outputs = scan(step, carry[layer], embedded)
                outputs.append(layer_outputs)
            logits = outputs[-1] @ params.fc_weights + params.fc_bias
            return logits
        else:
            outputs = []
            carry = [(h, c) for _ in range(self.num_layers)]
            decoder_input = jnp.array([tokenizer.bos_token_id])
            for _ in range(target_seq.shape[0]):
                embedded_input = params.embedding[decoder_input]
                for layer in range(self.num_layers):
                    carry[layer], output = self.lstm_cell(
                        carry[layer], embedded_input, params.decoder_lstm_weights[layer], training=False
                    )
                    embedded_input = output
                logits = output @ params.fc_weights + params.fc_bias
                logits = logits / temperature
                probs = jax.nn.softmax(logits)
                key = random.PRNGKey(int(time.time() * 1000))
                predicted_token = random.choice(key, jnp.arange(self.vocab_size), p=probs)
                outputs.append(predicted_token)
                decoder_input = predicted_token
            return jnp.stack(outputs)

    def forward(self, params, input_seq, target_seq=None, temperature=0.7, inference=False, training=True):
        h, c = self.encoder(input_seq, params, training)
        return self.decoder(target_seq[:, :-1] if not inference else target_seq, h, c, params, temperature, inference, training)

def loss_fn(params, model, input_seq, target_seq, pad_token_id):
    logits = model.forward(params, input_seq, target_seq, training=True)
    targets = target_seq[:, 1:]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    mask = targets != pad_token_id
    loss = -jnp.sum(jnp.sum(log_probs * jax.nn.one_hot(targets, model.vocab_size), axis=-1) * mask) / jnp.sum(mask)
    return loss

@alpa.parallelize
def train_step(params, model, batch, pad_token_id):
    input_seq, target_seq = batch
    grad_start = time.time()
    loss, grads = jax.value_and_grad(loss_fn)(params, model, input_seq, target_seq, pad_token_id)
    grad_time = time.time() - grad_start
    updated_params = jax.tree_map(lambda p, g: p - 0.001 * g, params, grads)
    return updated_params, loss, grad_time

def compute_bleu_score(model, params, test_batches, tokenizer, max_len=50):
    references = []
    hypotheses = []
    for i in range(0, len(test_batches), 128):
        batch = test_batches[i:i+128]
        input_seq = jnp.stack([b[0] for b in batch])
        target_seq = jnp.stack([b[1] for b in batch])
        output = model.forward(params, input_seq, target_seq, inference=True, temperature=0.7, training=False)
        for pred, tgt in zip(output, target_seq):
            pred_text = tokenizer.decode(pred, skip_special_tokens=True).split()
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True).split()
            hypotheses.append(pred_text)
            references.append([tgt_text])
    return corpus_bleu(references, hypotheses)

def train_and_test(model, train_batches, test_batches, num_gpus, epochs=10, batch_size=128):
    params = model
    process = psutil.Process()
    times = []
    throughputs = []
    mem_usages = []
    energies = []
    grad_times = []
    accuracies = []
    power_watts = 250 * num_gpus  # 250W per A100 GPU

    for epoch in range(epochs):
        start_time = time.time()
        total_samples = 0
        grad_time_epoch = 0
        for i in range(0, len(train_batches), batch_size):
            batch = train_batches[i:i+batch_size]
            input_seq = jnp.stack([b[0] for b in batch])
            target_seq = jnp.stack([b[1] for b in batch])
            params, loss, grad_time = train_step(params, model, (input_seq, target_seq), tokenizer.pad_token_id)
            grad_time_epoch += grad_time
            total_samples += len(batch)
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        throughputs.append(total_samples / epoch_time)
        mem_usages.append(process.memory_info().rss / 1024 ** 2)
        energies.append(power_watts * epoch_time)

        correct = 0
        total = 0
        for i in range(0, len(test_batches), batch_size):
            batch = test_batches[i:i+batch_size]
            input_seq = jnp.stack([b[0] for b in batch])
            target_seq = jnp.stack([b[1] for b in batch])
            output = model.forward(params, input_seq, target_seq, inference=True, training=False)
            predicted = output[:, 1:]
            target = target_seq[:, 1:]
            mask = target != tokenizer.pad_token_id
            correct += ((predicted == target) * mask).sum()
            total += mask.sum()
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)

        print(f"{num_gpus}-GPU Epoch {epoch + 1}, Time: {epoch_time:.2f}s, Throughput: {throughputs[-1]:.2f} samples/s, "
              f"Memory: {mem_usages[-1]:.2f}MB, Energy: {energies[-1]:.2f}J, Grad Time: {grad_times[-1]:.2f}s, "
              f"Accuracy: {accuracy:.4f}")
    return params, times, throughputs, mem_usages, energies, grad_times, accuracies

def chat_with_bot(model, params, tokenizer, max_len=50, temperature=0.7):
    latencies = []
    print("Starting conversation (type 'exit' or 'quit' to stop)...")
    while True:
        input_text = input("You: ")
        if input_text.lower() in ['exit', 'quit']:
            break
        start_time = time.time()
        try:
            input_tokens = tokenizer.encode(input_text, max_length=max_len, truncation=True, padding='max_length')
            input_tokens = jnp.array(input_tokens).reshape(1, -1)
            target_seq = jnp.zeros((1, max_len), dtype=jnp.int32)
            output = model.forward(params, input_tokens, target_seq, temperature=temperature, inference=True, training=False)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        latency = time.time() - start_time
        latencies.append(latency)
        print(f"Bot: {response}, Latency: {latency:.4f}s")
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average Inference Latency: {avg_latency:.4f}s")
    return avg_latency

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    train_data = RedditChatDataset('./train_input_tokens.npy', 
                                  './train_target_tokens.npy')
    test_data = RedditChatDataset('./test_input_tokens.npy', 
                                 './test_target_tokens.npy')
    train_batches = [(train_data[i][0], train_data[i][1]) for i in range(len(train_data))]
    test_batches = [(test_data[i][0], test_data[i][1]) for i in range(len(test_data))]

    key = random.PRNGKey(0)
    model = ChatbotModelJAX(tokenizer.vocab_size)

    # Single-GPU
    alpa.global_config.num_gpus = 1
    params_single, times_single, throughputs_single, mem_single, energies_single, grad_times_single, accuracies_single = train_and_test(
        model, train_batches, test_batches, num_gpus=1
    )

    # Multi-GPU
    alpa.global_config.num_gpus = 4
    params_multi, times_multi, throughputs_multi, mem_multi, energies_multi, grad_times_multi, accuracies_multi = train_and_test(
        model, train_batches, test_batches, num_gpus=4
    )

    # BLEU Score
    bleu_score = compute_bleu_score(model, params_multi, test_batches, tokenizer)
    print(f"BLEU Score (4-GPU): {bleu_score:.4f}")

    # Speedup and Efficiency
    avg_time_single = sum(times_single) / len(times_single)
    avg_time_multi = sum(times_multi) / len(times_multi)
    speedup = avg_time_single / avg_time_multi
    efficiency = speedup / 4
    print(f"Speedup (4-GPU): {speedup:.2f}x, Efficiency: {efficiency:.2f}")

    # Plotting
    epochs = range(1, 11)
    labels = ['Single-GPU', '4-GPU']
    all_times = [times_single, times_multi]
    all_mem = [mem_single, mem_multi]
    all_throughputs = [throughputs_single, throughputs_multi]
    all_energies = [energies_single, energies_multi]
    all_grad_times = [grad_times_single, grad_times_multi]
    all_accuracies = [accuracies_single, accuracies_multi]

    plt.figure(figsize=(15, 10))
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
    plt.ylabel('Energy (J)')
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
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./alpa_metrics_with_accuracy.png')

    avg_latency = chat_with_bot(model, params_multi, tokenizer)
    alpa.shutdown()