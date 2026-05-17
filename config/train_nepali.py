# Configuration for 16GB VRAM + 50k examples
dataset = 'nepali'
out_dir = 'out-nepali-50k'

# Model Architecture
n_layer = 12
n_head = 12
n_embd = 768
block_size = 512
vocab_size = 30522 # NepBERT

# Hyperparameters for 50k dataset
batch_size = 16            # Increased batch size for 16GB
gradient_accumulation_steps = 4 # Effective batch size = 16 * 4 = 64
learning_rate = 3e-4       # Standard LR for SFT
max_iters = 15000          # ~3-5 epochs over 50k samples
lr_decay_iters = 15000
min_lr = 3e-5
beta2 = 0.95

# Hardware settings
dtype = 'bfloat16'
compile = True             # Turn off if you encounter Windows C++ errors

log_interval = 10
eval_interval = 100
eval_iters = 150