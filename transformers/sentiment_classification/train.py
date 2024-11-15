"""
Script to finetune GPT-2
"""
import os
from typing import Dict
import click
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import loralib as lora
from reviewsDataset import reviewsDataset
from gpt import GPT
from gpt_utils import dynamic_padding
from gpt_config import GPTConfig
from train_config import TrainConfig


train_set = reviewsDataset(split="train")
tc = TrainConfig()
device = "mps"
writer = SummaryWriter(log_dir=tc.out_dir)

iter_num = 0
if tc.init_from == "resume":
    print(f"Initializing from checkpoint in {tc.out_dir}")
    if tc.use_lora:
        print(f"Initializing GPT-2 params from the original GPT-2 and LoRA params from {tc.lora_checkpoint}")
        model = GPT.from_pretrained(config=GPTConfig(binary_classification_head=True))
        ckpt_path = os.path.join(tc.out_dir, tc.lora_checkpoint)
        model.load_state_dict(torch.load(ckpt_path),strict=False)
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_params']
 
    else:
        ckpt_path = os.path.join(tc.out_dir, tc.checkpoint_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_params']
        model = GPT(GPTConfig(binary_classification_head=True,block_size=checkpoint_model_args.block_size))
        model.load_state_dict(checkpoint["model"])
    state_dict = checkpoint['model']
    iter_num = checkpoint["iter_num"]
elif tc.init_from == "gpt2":
    print(f"Initializing from GPT-2 parameters")
    model = GPT.from_pretrained(config=GPTConfig(binary_classification_head=True))

if tc.use_lora:
    lora.mark_only_lora_as_trainable(model)
if tc.block_size < model.config.block_size:
    model.crop_block_size(block_size=tc.block_size)

optimizer = model.configure_optimizers(tc.weight_decay, tc.learning_rate,(tc.beta1,tc.beta2),"mps")
scheduler = StepLR(optimizer,step_size=2000,gamma=0.1)
model.to(device=device)
if tc.init_from == "resume":
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(train_set: Dataset, val_set: Dataset) -> Dict[str,float]:
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
        optimizer.zero_grad(set_to_none=True)
        input_ids, attention_masks = batch["input_ids"].to(device), batch["attention_masks"].to(device)
        logits,loss = model(input_ids,attention_masks,target=batch["labels"].to(device))
        for param_group in optimizer.param_groups:
            param_group['lr'] = tc.learning_rate

        if tc.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),tc.grad_clip)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iter_num % tc.eval_interval == 0:
            losses = estimate_loss(train_set,val_set)
            print(f"Step: {iter_num}\n Train Loss: {losses['train']}\nValidation Loss: {losses['val']}")

            writer.add_scalar("Loss/train",losses["train"],iter_num)
            writer.add_scalar("Loss/val",losses["val"],iter_num)
            for name,param in model.named_parameters():
                writer.add_histogram(name, param, iter_num)
                # writer.add_histogram(f"{name}/grad",param.grad,iter_num)

            if losses["val"] < best_val_loss or tc.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {"model": lora.lora_state_dict(model),
                                  "model_params": tc,
                                  "optimizer":optimizer.state_dict(),
                                  "iter_num": iter_num,
                                  "best_val_loss": best_val_loss,
                                  "config": tc}
                
                    print(f"Saving checkpoint to {tc.out_dir}")
                    if not os.path.exists(tc.out_dir):
                        os.makedirs(tc.out_dir)
                    if tc.use_lora:
                        output_path = os.path.join(tc.out_dir,tc.lora_checkpoint)
                    else:
                        output_path = os.path.join(tc.out_dir,tc.checkpoint_name)
                    torch.save(checkpoint,output_path)

        iter_num += 1