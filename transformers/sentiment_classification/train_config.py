"""
Train Config
"""
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # I/O
    out_dir:str = "out"
    checkpoint_name: str = "ckpt_train.pt"
    eval_interval:int = 500
    eval_iters:int = 20
    eval_only:bool = False
    always_save_checkpoint:bool = True
    init_from:str = "gpt2" # 'gpt2' or 'resume' - it will resume from the latest checkpoint

    # data
    batch_size:int = 2

    # AdamW optimizer
    learning_rate:float = 6e-4
    max_iters:int = 30000
    weight_decay:float = 1e-1
    beta1:float = 0.9
    beta2:float = 0.95
    grad_clip:float = 1.0

    # Learning Rate scheduler : StepLR
    step_size:int = 10000
    warmup_iters:int = 500
    lr_decay_iters:int = 30000
    min_lr: float = 6e-5

    #device
    device:str = "mps"