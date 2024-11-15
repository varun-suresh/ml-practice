from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    embedding_size: int = 768
    binary_classification_head: bool = False

    # LoRA params
    use_lora = True
    r = 8
