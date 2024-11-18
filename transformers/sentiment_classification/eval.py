"""
Evaluate fine-tuned GPT-2 on IMDb movie reviews
"""

import click
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from reviewsDataset import reviewsDataset
from gpt_utils import dynamic_padding
from gpt_config import GPTConfig
from gpt import GPT

device = "mps"
# Zero Shot Eval
config = GPTConfig(use_lora=False)
model = GPT.from_pretrained()
model.crop_block_size(config.block_size)

# config = GPTConfig(binary_classification_head=True,block_size=128)
# if config.use_lora:
#     checkpoint = torch.load("out/ckpt_lora.pt")
#     model = GPT.from_pretrained(config=GPTConfig(binary_classification_head=True))
#     model.crop_block_size(128)
#     model.load_state_dict(checkpoint["model"],strict=False)
# else:
#     checkpoint = torch.load("out/ckpt.pt")
#     model = GPT(config)
#     model.load_state_dict(checkpoint["model"])

model.to(device=device)
model.eval()

@click.command()
@click.option("--split",default="test",type=click.Choice(["test","train"]))
@click.option("--batch_size",default=2)
@click.option("--max_length",default=128)
@click.option("--results_fname",default="zero_shot.txt")
@click.option("--subset",default=True)
def run_inference(split,batch_size,max_length,results_fname,subset):
    rd = reviewsDataset(split=split,max_length=max_length)
    if subset:
        subset_range = torch.arange(0,len(rd),10)
        dl = DataLoader(torch.utils.data.Subset(rd,subset_range),batch_size=batch_size,collate_fn=dynamic_padding)
    else:
        dl = DataLoader(rd,batch_size=batch_size,collate_fn=dynamic_padding)
    
    results_file = open(results_fname,"w")
    results_file.write(f"filename,length,label,prediction,pos_probability,neg_probability\n")
    for batch in tqdm(dl):
        with torch.no_grad():
            logits, _ = model(batch["input_ids"].to(device),batch["attention_masks"].to(device))
            sentiment_idx = rd.get_pos_neg_indices()
            for i, fname in enumerate(batch["fpaths"]):
                pos = logits[i,sentiment_idx["positive"]]
                neg = logits[i,sentiment_idx["negative"]]
                if pos > neg:
                    prediction = 1
                else:
                    prediction = 0
 
                results_file.write(f"{fname},{batch['lengths'][i]},{batch['labels'][i]},{prediction},{pos},{neg}\n")
if __name__ == "__main__":
    run_inference()