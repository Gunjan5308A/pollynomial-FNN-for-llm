import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# 1. Load data and tokenizer
dataset = load_dataset("himalaya-ai/nepali-sft-dataset", split='train[:50000]')
tokenizer = AutoTokenizer.from_pretrained("Rajan/NepaliBERT")

# Explicitly define a token id to act as EOS since BERT tokenizers lack one.
# Rajan/NepaliBERT uses 3 for [SEP], which works perfectly as an end marker.
EOS_TOKEN_ID = 3 

def process(example):
    conversations = example.get('conversations', [])
    system_prompt = ""
    instruction = ""
    output = ""
    
    for msg in conversations:
        role = msg.get('from')
        value = msg.get('value', '').strip()
        if role == 'system':
            system_prompt = value
        elif role == 'human':
            instruction = value
        elif role == 'gpt':
            output = value
            
    # If there is no instruction or response, skip it entirely
    if not instruction or not output:
        return {'ids': [], 'len': 0}
        
    # Format text cleanly
    full_text = ""
    if system_prompt:
        full_text += f"### System:\n{system_prompt}\n"
    full_text += f"### Instruction:\n{instruction}\n"
    full_text += f"### Response:\n{output}"
    
    # Encode text to IDs, then manually append the EOS ID
    out = tokenizer.encode(full_text, add_special_tokens=False)
    out.append(EOS_TOKEN_ID)
    
    return {'ids': out, 'len': len(out)}


# 2. Split and Map
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
split_dataset['val'] = split_dataset.pop('test')

tokenized = split_dataset.map(
    process,
    remove_columns=dataset.column_names,
    desc="Tokenizing 50k Nepali examples",
    num_proc=4 # Setting to 4 is safer for lower-end CPUs to prevent memory locks
)

# Filter out empty examples that we skipped
tokenized = tokenized.filter(lambda x: x['len'] > 0)

# 3. Write to binary
# 3. Write to binary (Fixed Alignment Logic)
for split, dset in tokenized.items():
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    
    # Calculate total tokens
    total_tokens = sum(dset['len'])
    
    # CRITICAL FIX: If the total token count is odd, make it even.
    # An odd number of uint16 tokens will always cause a memmap size mismatch error.
    if total_tokens % 2 != 0:
        print(f"Warning: {split} had an odd token count ({total_tokens}). Adjusting to even.")
        total_tokens -= 1 

    # Create the memory-mapped file with the guaranteed even shape
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(total_tokens,))
    
    idx = 0
    for example in tqdm(dset, desc=f"Writing {filename}"):
        # Check if adding this example exceeds our adjusted even boundary
        if idx + example['len'] > total_tokens:
            available_space = total_tokens - idx
            arr[idx : idx + available_space] = example['ids'][:available_space]
            break
            
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
        
    arr.flush()

print("\nData preparation complete safely! Your binaries are now perfectly aligned.")
