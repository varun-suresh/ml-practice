"""
Train Config
"""
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # I/O
    out_dir = "out"
    eval_interval = 2000
    log_interval = 1
    eval_iters = 20
    eval_only = False
    always_save_checkpoint = True
    init_from = "resume" # or 'resume' - it will resume from the latest checkpoint

    # data
    dataset = "imdb"
    batch_size = 2
    block_size = 128
    n_epochs = 3

    # AdamW optimizer
    learning_rate = 6e-4
    max_iters = 100000
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0

