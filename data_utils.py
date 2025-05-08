import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dailydialog_data(tokenizer, max_len=30, limit=1000):
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

    np.save('train_input_tokens.npy', np.array(input_tokens_list[:int(0.8 * len(input_tokens_list))]))
    np.save('train_target_tokens.npy', np.array(target_tokens_list[:int(0.8 * len(target_tokens_list))]))
    np.save('test_input_tokens.npy', np.array(input_tokens_list[int(0.8 * len(input_tokens_list)):]))
    np.save('test_target_tokens.npy', np.array(target_tokens_list[int(0.8 * len(target_tokens_list)):]))
