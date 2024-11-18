import torch
from torch.nn.utils.rnn import pad_sequence

def dynamic_padding(data):
    inputs = [item["input_ids"] for item in data]
    attention_masks = [item["attention_mask"] for item in data]
    labels = [item["label"] for item in data]
    label_idxs = [item["label_idx"] for item in data]
    fpaths = [item["fpath"] for item in data]
    lengths = [item["length"] for item in data]
    inputs_padded = pad_sequence(inputs, batch_first=True,padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks,batch_first=True,padding_value=0)
    labels = torch.tensor(labels,dtype=torch.float)
    label_idxs = torch.tensor(label_idxs,dtype=torch.float)
    lengths = torch.tensor(lengths)
    return {"input_ids": inputs_padded, "attention_masks":attention_masks_padded, "labels":labels, "fpaths": fpaths, "lengths": lengths, "label_idxs": label_idxs}

