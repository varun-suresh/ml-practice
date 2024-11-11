from tqdm import tqdm
import torch
import click
from sentiment_classification.reviewsDataset import reviewsDataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
# Add directory to PYTHONPATH
from gpt import GPT

device = "mps"
model = GPT.from_pretrained(model_type="gpt2")
model.to(device)
model.eval()


def dynamic_padding(data):
    inputs = [item["input_ids"] for item in data]
    attention_masks = [item["attention_mask"] for item in data]
    labels = [item["label"] for item in data]
    fpaths = [item["fpath"] for item in data]
    lengths = [item["length"] for item in data]
    inputs_padded = pad_sequence(inputs, batch_first=True,padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks,batch_first=True,padding_value=0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return {"input_ids": inputs_padded, "attention_masks":attention_masks_padded, "labels":labels, "fpaths": fpaths, "lengths": lengths}

@click.command()
@click.option("--split",default="test",type=click.Choice(["test","train"]),help="Dataset split to use")
@click.option("--batch_size",default=2,help="No of sequences to process at a time")
@click.option("--max_length",default=128,help="Maximum seqeunce length for GPT-2")
@click.option("--results_fname",default="zero_shot.txt",help="Results file name")
@click.option("--subset",default=True)
def run_inference(split,batch_size,max_length,results_fname,subset):
    rd = reviewsDataset(split=split,max_length=max_length)
    if subset:
        subset_range = torch.arange(0,len(rd),10)
        dl = DataLoader(torch.utils.data.Subset(rd,subset_range),batch_size=batch_size,collate_fn=dynamic_padding)
    else:
        dl = DataLoader(rd,batch_size=batch_size,collate_fn=dynamic_padding)

    positive_sentiment_id = 33733
    negative_sentiment_id = 36183

    results_file = open(results_fname,"w")
    results_file.write("filename,length,label,prediction\n")
    for batch in tqdm(dl):
        with torch.no_grad():
            input_ids, attention_masks = batch["input_ids"].to(device), batch["attention_masks"].to(device)
            logits, _ = model(input_ids,attention_masks)
            sequence_lengths = torch.sum(attention_masks,dim=1,dtype=int) - 1
            batch_indices = torch.arange(logits.size(0))
            selected_logits = logits[batch_indices,sequence_lengths]
            positive_scores = selected_logits[torch.arange(selected_logits.size(0)),positive_sentiment_id*torch.ones_like(batch_indices)]
            negative_scores = selected_logits[torch.arange(selected_logits.size(0)),negative_sentiment_id*torch.ones_like(batch_indices)]
            predicted_labels = torch.tensor(positive_scores > negative_scores,dtype=torch.int)
            for i,fname in enumerate(batch["fpaths"]):
                results_file.write(f"{fname},{batch['lengths'][i]},{batch['labels'][i]},{predicted_labels[i]}\n")

if __name__ == "__main__":
    run_inference()