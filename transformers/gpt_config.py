from dataclasses import dataclass

MODELS = {
    "gpt2": dict(n_layers=12,n_heads=12,embedding_size=768),
    "gpt2-medium": dict(n_layers=24,n_heads=16,embedding_size=1024),
    "gpt2-large": dict(n_layers=36,n_heads=20,embedding_size=1024),
    "gpt2-xl": dict(n_layers=48,n_heads=25,embedding_size=1600),
}

@dataclass
class GPTConfigDefault:
    vocab_size: int = 50257
    block_size: int = 1024
    binary_classification_head: bool = False
    model_type:str = "gpt2"
    dropout: float = 0.0
    debug:bool = False
    freeze_layers:int = 0

    def __post_init__(self):
        if self.model_type not in MODELS:
            raise ValueError(f"Invalid model type {self.model_type}")
        for k,v in MODELS[self.model_type].items():
            setattr(self,k,v)
@dataclass
class GPTConfig(GPTConfigDefault):
    block_size:int = 128
    model_type: str 
    # Load from a checkpoint
    load_from_checkpoint : bool = False
    checkpoint_path : str = ""

    # LoRA params
    use_lora:bool = True
    r:int = 8
    # No of transformer blocks to freeze when setting up for training
    # Note: This setting should ideally be in the train config. Putting it here because when freeze_layers is used in conjunction
    # with LoRA, I want to only setup LoRA params for the layers that aren't frozen.
    freeze_layers: int = 10
    # Use a binary classification head
    binary_classification_head:bool = False

    # Regularizaztion
    dropout: float = 0.1

    # For debugging
    debug:bool = False
