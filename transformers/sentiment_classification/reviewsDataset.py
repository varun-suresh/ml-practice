# IMDb reviews dataloader
import os
import torch
import tiktoken
from torch.utils.data import Dataset
from typing import List

class reviewsDataset(Dataset):
    """
    Contains the IMDb reviews
    """
    def __init__(self, split: str, cache_dir: str = "/Users/varun/Downloads/aclImdb",max_length=128):
        assert split in {'train', 'test'}
        self.split = split
        self.cache_dir = cache_dir
        self.enc = tiktoken.get_encoding("gpt2")
        self.max_length = max_length
        self.label_mapping = {0: "Negative", 1: "Positive"}
        self.summary_stats = {}
        self.data = []
        pos_dir = os.path.join(self.cache_dir,split,"pos")
        neg_dir = os.path.join(self.cache_dir,split,"neg")
        self._prepare(pos_dir,1)
        self._prepare(neg_dir,0)

    def _prepare(self, path: str, label:int):
        count = 0
        for fname in os.listdir(path):
            count += 1
            self.data.append([os.path.join(path,fname),label])
        self.summary_stats[label] = count

    def __len__(self):
        """
        Returns the number of examples in the train/test set as specified while initializing
        """
        return len(self.data)

    def encode(self, s: str):
        return self.enc.encode(s, allowed_special={"<|endoftext|>"})

    def __getitem__(self, idx: int):
        fpath, label = self.data[idx]

        review = open(fpath).read().replace("<br />","")
        review = f"Review: The movie was awesome. Sentiment: Positive. Review: The performances were disappointing. Sentiment: Negative. Review: {review} Sentiment:"
        review_ids_orig = torch.tensor(self.encode(review))
        review_ids = review_ids_orig[-self.max_length:]
        attention_mask = torch.ones(review_ids.size(),dtype=torch.bool)
        return {
            "input_ids": review_ids,
            "length": review_ids_orig.size(0),
            "attention_mask": attention_mask,
            "label": label,
            "fpath": fpath,
        }

    def summary(self):
        """
        Prints summary statistics for the dataset
        """
        print(f"Total reviews in {self.split} is {len(self.data)}")
        print(f"Positive reviews in {self.split} is {self.summary_stats[1]}")
        print(f"Negative reviews in {self.split} is {self.summary_stats[0]}")

if __name__ == "__main__":
    rd = reviewsDataset("train")
    rd.summary()