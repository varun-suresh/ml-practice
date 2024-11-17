"""
Train Config
"""
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # I/O
    out_dir = "out"
    checkpoint_name = "ckpt_train.pt"
    eval_interval = 500
    eval_iters = 20
    eval_only = False
    always_save_checkpoint = True
    init_from = "gpt2" # 'gpt2' or 'resume' - it will resume from the latest checkpoint

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

    # Learning Rate scheduler : StepLR
    step_size = 10000


    #device
    device = "mps"