# Imports
import math
from torch import nn
import torch
from torch.nn import functional as F
from transformer_config import TransformerConfig

# Attention layer
class ScaledDotProductAttention(nn.Module):
  def __init__(self,config):
    self.config = config
    super(ScaledDotProductAttention, self).__init__()
    self.attn = nn.Linear(self.config.d_model, 3*self.config.d_model)
    self.register_buffer("mask", torch.tril(torch.ones(config.d_block, config.d_block)).view(1,config.d_block,config.d_block))

  def forward(self, x, mask=False):
    
    B,T,C = x.shape

    k,q,v = self.attn(x).split(self.config.d_model,dim=2)
    kq = torch.matmul(k, torch.transpose(q, -1,-2)) / math.sqrt(self.config.d_model)
    if mask:
      kq = kq.masked_fill(self.mask[:,:T,:T] == 0, float('-inf'))
    kq_softmax = kq.softmax(-1)
    out = torch.matmul(kq_softmax, v)
    return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
      self.config = config
      assert self.config.d_model % self.config.n_heads == 0
      super(MultiHeadedAttention, self).__init__()
      self.attn = nn.Linear(self.config.d_model, 3*self.config.d_model)
      self.register_buffer("mask", torch.tril(torch.ones(self.config.d_block, self.config.d_block).view(1,1,self.config.d_block, self.config.d_block)))
      self.n_heads = config.n_heads

    def forward(self,x, mask=False):
        B,T,C = x.shape # Batch Size, Block Size, Embedding Size
        k,q,v = self.attn(x).split(self.config.d_model, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)

        kq = (k @ q.transpose(-2,-1)) / math.sqrt(k.shape[-1])
        if mask:
           kq = kq.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        kq_softmax = kq.softmax(-1)
        out = kq_softmax @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return out



# Feed Forward Block
class FeedForward(nn.Module):
  def __init__(self, config):
    super(FeedForward, self).__init__()
    self.fc1 = nn.Linear(config.d_model, config.d_ff)
    self.fc2 = nn.Linear(config.d_ff, config.d_model)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class TransformerBlock(nn.Module):
  def __init__(self,config, decoder=True):
    super(TransformerBlock, self).__init__()
    self.ln_1 = nn.LayerNorm(config.d_model)
    self.ln_2 = nn.LayerNorm(config.d_model)
    if config.attention == "mh":
       self.attn = MultiHeadedAttention(config)
    else:
        self.attn = ScaledDotProductAttention(config)
    self.ffn = FeedForward(config)
    self.decoder = decoder


  def forward(self, x):
    x = x + self.attn(self.ln_1(x), mask=self.decoder)
    x = x + self.ffn(self.ln_2(x))
    return x
  
class Transformer(nn.Module):
    def __init__(self, config):
      super(Transformer, self).__init__()
      self.config = config
      self.layers = nn.ModuleDict({
          "embedding" : nn.Embedding(self.config.d_vocab, self.config.d_model),
          "positional_encoding": nn.Embedding(self.config.d_block, self.config.d_model),
          "decoder": nn.ModuleList([TransformerBlock(self.config, decoder=True) for _ in range(self.config.n_decoders)]),
          "logits": nn.Linear(self.config.d_model, self.config.d_vocab),
      })

    def forward(self,idx, targets=None):
        device = idx.device
        b,t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        emb = self.layers["embedding"](idx)
        pe = self.layers["positional_encoding"](pos)
        x = emb + pe
        x = x.view(b,t,self.config.d_model)
        for layer in self.layers["decoder"]:
            x = layer(x)
        logits = self.layers["logits"](x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, config):
       optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
       return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.2):

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.d_block else idx[:, -self.config.d_block:]
            # print(idx_cond)
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:] / temperature
            probs = F.softmax(logits, dim=-1)
            # print(probs)
            # _, idx_next = torch.topk(probs,k=1,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next),dim=1)
        return idx

if __name__ == "__main__":
    config = TransformerConfig(d_model=512, d_ff=2048,d_block=11,d_vocab=3)
    t = Transformer(config)
    input = torch.tensor([[0,1,2]])
    print(t(input))
    # for param in t.parameters():
    #     print(param.shape)
    #     # out, loss = t(torch.tensor([0,1,2]))
    #     idx = t.generate(torch.tensor([[1,2,3]]),max_new_tokens=5)
    #     print(idx)