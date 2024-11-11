from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import math


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.config = config
        assert self.config.embedding_size % self.config.n_heads == 0
        self.c_attn = nn.Linear(
            self.config.embedding_size, 3 * self.config.embedding_size,
        )
        self.c_proj = nn.Linear(self.config.embedding_size, self.config.embedding_size)

    def forward(self, x, attention_mask):
    # def forward(self,x):
        B, T, C = x.shape  # Batch Size, Block Size/ Sequence Length, Embedding Size
        q, k, v = self.c_attn(x).split(self.config.embedding_size, dim=2)
        q = q.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2)
        k = k.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2)
        v = v.view(
            B, T, self.config.n_heads, self.config.embedding_size // self.config.n_heads
        ).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask.view(B,1,1,-1),
        )
        # y = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.c_fc = nn.Linear(config.embedding_size, 4 * config.embedding_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embedding_size, config.embedding_size)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_size)
        self.attn = MultiHeadedAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_size)
        self.mlp = FeedForward(config)

    def forward(self, x, attention_mask):
    # def forward(self, x):
        x += self.attn(self.ln_1(x),attention_mask)
        # x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    embedding_size: int = 768


class GPT(nn.Module):
    """
    Define the GPT-2 architecture and the ability to load the pretrained model
    """

    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.config.vocab_size, self.config.embedding_size),
                "wpe": nn.Embedding(self.config.block_size, self.config.embedding_size),
                "h": nn.ModuleList(
                    TransformerBlock(self.config) for _ in range(self.config.n_layers)
                ),
                "ln_f": nn.LayerNorm(config.embedding_size),
            }
        )

        self.lm_head = nn.Linear(
            self.config.embedding_size, self.config.vocab_size, bias=False
        )
        self.transformer.wte.weight = (self.lm_head.weight)
        # Init weights:
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )


        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self,non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx,attention_mask,target=None):
    # def forward(self, idx, target=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Sequence length {t} is larger than the block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            # x = block(x)
            x = block(x,attention_mask)
        x = self.transformer.ln_f(x)

        if target is not None:

            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x)
            # logits = self.lm_head(x[:,[-1],:])
            loss = None
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        """
        Downloads the Hugging Face model and copies the pre-trained weights on to the model defined here.
        """
        from transformers import GPT2LMHeadModel

        print(f"Loading pre-trained weights for {model_type}")
        config = GPTConfig()
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # Init a hugging face transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # print(f"Hugging Face {k}: {sd_hf[k].shape}, Custom : {sd[k].shape}")
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=1.0, top_k=None):
        """
        Take a conditioning sequence and generate max_new_tokens number of tokes. Predictions are fed back in to the model each time
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]/temp
             # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        
        return idx
    
if __name__ == "__main__":
    config = GPTConfig()
    model = GPT.from_pretrained()
