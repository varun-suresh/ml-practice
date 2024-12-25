from dataclasses import dataclass

@dataclass
class GPTConfigDefault:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    embedding_size: int = 768
    binary_classification_head: bool = False
    dropout: float = 0.0
    debug:bool = False

@dataclass
class GPTConfig(GPTConfigDefault):
    block_size:int = 128
    
    # Load from a checkpoint
    load_from_checkpoint : bool = False
    checkpoint_path : str = ""
    # LoRA params
    use_lora:bool = True
    r:int = 8
    
    # Use a binary classification head
    binary_classification_head:bool = True

    # Regularizaztion
    dropout: float = 0.2

    # For debugging
    debug:bool = False