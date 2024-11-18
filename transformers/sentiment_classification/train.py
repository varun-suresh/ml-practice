"""
Script to finetune GPT-2
"""
import os
from typing import Dict
import click
from tqdm import tqdm
from dataclasses import asdict
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import loralib as lora
from reviewsDataset import reviewsDataset
from gpt import GPT
from gpt_utils import dynamic_padding
from gpt_config import GPTConfig, GPTConfigDefault
from train_config import TrainConfig

torch.manual_seed(1367)

class Trainer:
    def __init__(self,train_set: Dataset,val_set: Dataset,train_config:TrainConfig,model_config:GPTConfig):
        self.train_set = train_set
        self.val_set = val_set
        self.train_config = train_config
        self.model_config = model_config
        self.writer = SummaryWriter(log_dir=self.train_config.out_dir)
        self.iter_num = 0

    def load_model(self):

        self.model = GPT.from_pretrained()
        if self.model_config.use_lora:
            self.model.setup_lora(self.model_config.r)
        if self.model_config.block_size < GPTConfigDefault().block_size:
            self.model.crop_block_size(self.model_config.block_size)
        if self.train_config.init_from == "resume":
            ckpt_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name)
            print(f"Resuming training from {ckpt_path}")
            self.ckpt = torch.load(ckpt_path,map_location=self.train_config.device)
            try:
                self.model.load_state_dict(checkpoint["model"])
            except Exception as e:
                print(f"Check the model config (using or not using LoRA, block size). The exception was {e}")
        self.model.to(self.train_config.device)

    def load_scheduler_optimizer(self):
        self.optimizer = self.model.configure_optimizers(self.train_config.weight_decay, 
                                                self.train_config.learning_rate,
                                                (self.train_config.beta1,self.train_config.beta2),
                                                self.train_config.device)
        self.scheduler = StepLR(self.optimizer,
                        step_size=self.train_config.step_size,
                        gamma=0.1)
        if self.train_config.init_from =="resume":
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.scheduler.load_state_dict(self.ckpt['scheduler'])
    
 
    def train(self):

        self.load_model()
        self.load_scheduler_optimizer()
        if self.train_config.init_from == "resume":
            start_iter = self.ckpt["iter_num"]
            best_val_loss = self.ckpt["best_val_loss"]
        else:
            start_iter = 0
            best_val_loss = 1e9

        dl = DataLoader(self.train_set, 
                        batch_size=self.train_config.batch_size,
                        collate_fn=dynamic_padding,
                        shuffle=True)
        for iter_num in tqdm(range(start_iter,self.train_config.max_iters)):
            batch = next(iter(dl))
            self.optimizer.zero_grad(set_to_none=True)
            logits,loss = self.model(batch["input_ids"].to(self.train_config.device),
                                    batch["attention_masks"].to(self.train_config.device),
                                    target=batch["label_idxs"].to(self.train_config.device))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.train_config.learning_rate

            if self.train_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.train_config.grad_clip)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"Step: {iter_num}\n Train Loss: {losses['train']}\nValidation Loss: {losses['val']}")

                self.writer.add_scalar("Loss/train",losses["train"],iter_num)
                self.writer.add_scalar("Loss/val",losses["val"],iter_num)
                for name,param in self.model.named_parameters():
                    if param.requires_grad:
                        self.writer.add_histogram(name, param, iter_num)
                        self.writer.add_histogram(f"{name}/grad",param.grad,iter_num)

                if losses["val"] < best_val_loss or self.train_config.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        ckpt = {"model": self.model.state_dict(),
                                    "train_config": asdict(self.train_config),
                                    "optimizer":self.optimizer.state_dict(),
                                    "scheduler": self.scheduler.state_dict(),
                                    "iter_num": iter_num,
                                    "best_val_loss": best_val_loss,
                                }
                        output_path = os.path.join(self.train_config.out_dir,self.train_config.checkpoint_name) 
                        print(f"Saving checkpoint to {output_path}") 
                        if not os.path.exists(self.train_config.out_dir):
                            os.makedirs(self.train_config.out_dir)
                        torch.save(ckpt,output_path)


    @torch.no_grad()
    def estimate_loss(self) -> Dict[str,float]:
        self.model.eval()
        train_dl = DataLoader(self.train_set,
                            batch_size=self.train_config.batch_size,
                            collate_fn=dynamic_padding,
                            shuffle=True)
        val_dl = DataLoader(self.val_set,
                            batch_size=self.train_config.batch_size,
                            collate_fn=dynamic_padding,
                            shuffle=True)
        train_loss = torch.zeros(self.train_config.eval_iters)
        val_loss = torch.zeros(self.train_config.eval_iters)
        for i in range(self.train_config.eval_iters):
            train_batch = next(iter(train_dl))
            val_batch = next(iter(val_dl))
            _, train_loss[i] = self.model(train_batch['input_ids'].to(self.train_config.device), 
                                    train_batch['attention_masks'].to(self.train_config.device),
                                    target=train_batch['labels'].to(self.train_config.device))
            _, val_loss[i] = self.model(val_batch['input_ids'].to(self.train_config.device),
                               val_batch['attention_masks'].to(self.train_config.device),
                               target=val_batch['labels'].to(self.train_config.device))
        losses = {}
        losses["train"] = train_loss.mean()
        losses["val"] = val_loss.mean()
        self.model.train()
        return losses

if __name__ == "__main__":
    train_config = TrainConfig()
    model_config = GPTConfig()
    rd = reviewsDataset(split="train",max_length=train_config.block_size)
    train_set, val_set = torch.utils.data.random_split(rd,[0.85,0.15])
    trainer = Trainer(train_set,val_set,train_config,model_config)
    trainer.train()