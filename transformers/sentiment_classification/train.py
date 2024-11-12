"""
Script to finetune GPT-2
"""
import os
import click
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from reviewsDataset import reviewsDataset
from gpt import GPT
from gpt_utils import dynamic_padding
from gpt_config import GPTConfig
from train_config import TrainConfig


config = GPTConfig(binary_classification_head=True)
train_set = reviewsDataset(split="train")
tc = TrainConfig()
device = "mps"

iter_num = 0
if tc.init_from == "resume":
    print(f"Initializing from checkpoint in {tc.out_dir}")
    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # checkpoint_model_args = checkpoint['model_args']
    model = GPT(config)
    state_dict = checkpoint['model']
    iter_num = checkpoint["iter_num"]
elif tc.init_from == "gpt2":
    print(f"Initializing from GPT-2 parameters")
    model = GPT.from_pretrained(config=config)

if tc.block_size < model.config.block_size:
    model.crop_block_size(block_size=tc.block_size)

optimizer = model.configure_optimizers(tc.weight_decay, tc.learning_rate,(tc.beta1,tc.beta2),"mps")
model.to(device=device)
if tc.init_from == "resume":
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(train_set, val_set):
    model.eval()
    train_dl = DataLoader(train_set,batch_size=tc.batch_size,collate_fn=dynamic_padding,shuffle=True)
    val_dl = DataLoader(val_set,batch_size=tc.batch_size,collate_fn=dynamic_padding,shuffle=True)
    train_loss = torch.zeros(tc.eval_iters)
    val_loss = torch.zeros(tc.eval_iters) 
    for i in range(tc.eval_iters):
        train_batch = next(iter(train_dl))
        val_batch = next(iter(val_dl))
        _, train_loss[i] = model(train_batch['input_ids'].to(device), train_batch['attention_masks'].to(device),target=train_batch['labels'].to(device))
        _, val_loss[i] = model(val_batch['input_ids'].to(device),
                               val_batch['attention_masks'].to(device),
                               target=val_batch['labels'].to(device))
    out = {}
    out["train"] = train_loss.mean()
    out["val"] = val_loss.mean()
    model.train()
    return out


rd = reviewsDataset(split="train",max_length=tc.block_size)
train_set, val_set = torch.utils.data.random_split(rd,[0.85,0.15])
dl = DataLoader(train_set, batch_size=tc.batch_size,collate_fn=dynamic_padding,shuffle=True)
best_val_loss = 1e9
for epoch in range(tc.n_epochs):
    for batch in tqdm(dl):
        input_ids, attention_masks = batch["input_ids"].to(device), batch["attention_masks"].to(device)
        logits,loss = model(input_ids,attention_masks,target=batch["labels"].to(device))
        for param_group in optimizer.param_groups:
            param_group['lr'] = tc.learning_rate

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_num % tc.eval_interval == 0:
            losses = estimate_loss(train_set,val_set)
            print(f"Step: {iter_num}\n Train Loss: {losses['train']}\nValidation Loss: {losses['val']}")
        
            if losses["val"] < best_val_loss or tc.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {"model": model.state_dict(),
                                  "optimizer":optimizer.state_dict(),
                                  "iter_num": iter_num,
                                  "best_val_loss": best_val_loss,
                                  "config": tc}
                
                    print(f"Saving checkpoint to {tc.out_dir}")
                    if not os.path.exists(tc.out_dir):
                        os.makedirs(tc.out_dir)
                    torch.save(checkpoint,os.path.join(tc.out_dir,"ckpt.pt"))
        iter_num += 1