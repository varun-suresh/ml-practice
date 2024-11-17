from dataclasses import dataclass

@dataclass
class GPTConfigDefault:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    embedding_size: int = 768

@dataclass
class GPTConfig(GPTConfigDefault):
    block_size = 128
    # LoRA params
    use_lora = True
    r = 8