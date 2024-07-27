
class TransformerConfig:
    def __init__(self, d_model, d_ff, d_block, d_vocab, n_decoders=6,attention="mh",n_heads=8):
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_block = d_block
        self.d_vocab = d_vocab
        self.n_decoders = n_decoders
        self.attention = attention
        self.n_heads = n_heads
    
    def __repr__(self) -> str:
        return f"Model Dimension: {self.d_model}\nFeed Forward: {self.d_ff}\nBlock Size: {self.d_block}\nVocab size: {self.d_vocab}\nNo of Decoders: {self.n_decoders}\nNumber of heads: {self.n_heads}"
