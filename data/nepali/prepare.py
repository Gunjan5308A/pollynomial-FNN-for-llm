import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# 1. Load exactly 50k examples
# Note: Ensure the split name is 'train' or 'train[:50000]'
dataset = load_dataset("himalaya-ai/nepali-sft-dataset", split='train[:50000]')
tokenizer = AutoTokenizer.from_pretrained("Rajan/NepaliBERT")

def process(example):
    # Himalaya-SFT usually has 'instruction', 'input', 'output'
    # We combine them into a single string for the causal LLM to learn
    instruction = example.get('instruction', '')
    inp = example.get('input', '')
    output = example.get('output', '')
    
    # Format: instruction + context -> response
    full_text = f"### Instruction:\n{instruction}\n"
    if inp:
        full_text += f"### Input:\n{inp}\n"
    full_text += f"### Response:\n{output}{tokenizer.eos_token}"
    
    out = tokenizer.encode(full_text, add_special_tokens=False)
    return {'ids': out, 'len': len(out)}

# 2. Tokenize and Split
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
split_dataset['val'] = split_dataset.pop('test')

tokenized = split_dataset.map(
    process,
    remove_columns=dataset.column_names,
    desc="Tokenizing 50k Nepali examples",
    num_proc=8
)

# 3. Write to binary
for split, dset in tokenized.items():
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    # Using uint32 is safer for large datasets or large vocabularies
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(sum(dset['len']),))
    
    idx = 0
    for example in tqdm(dset, desc=f"Writing {filename}"):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()