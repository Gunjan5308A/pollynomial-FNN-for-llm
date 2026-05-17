# Config for a 64M Parameter Nepali Model on 4GB VRAM
dataset = 'nepali'         # Points to data/nepali/
out_dir = 'out-nepali-64m'

# Architecture
n_layer = 8
n_head = 8
n_embd = 512
block_size = 256           # 256 tokens is perfect for short SFT Q&A
vocab_size = 30522         # Matches NepBERT exactly

# Memory Optimization for RTX 3050 (4GB)
batch_size = 2             # Micro-batch size 
gradient_accumulation_steps = 32 # Effective batch size = 2 * 32 = 64

# Hyperparameters for 50k SFT dataset
learning_rate = 6e-4     # Slightly higher LR for a smaller model 6e-4
max_iters = 12000        # Gives ~3 passes over your 50k dataset #12000
lr_decay_iters = 12000 #12000
min_lr = 6e-5 #6e-5
beta2 = 0.95
weight_decay = 0.1

# Hardware settings for Windows
dtype = 'bfloat16'         # Keeps memory footprint small
compile = False            # Set to False on Windows to avoid startup VRAM spikes
log_interval = 10
eval_interval = 100
eval_iters = 120