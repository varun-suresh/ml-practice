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
        # prompt_prefix = "Review: The movie was awesome. Sentiment: Positive. Review: The movie was disappointing. Sentiment: Negative. Review:"
        prompt_prefix = "Review:"
        self.prompt_prefix_ids = self.encode(prompt_prefix)
        prompt_suffix = "Sentiment:"
        self.prompt_suffix_ids = self.encode(prompt_suffix)
        self.max_length = max_length
        self.summary_stats = {}
        self.data = []
        pos_dir = os.path.join(self.cache_dir,split,"pos")
        neg_dir = os.path.join(self.cache_dir,split,"neg")
        self.pos_index = self.encode(" Positive")[0]
        self.neg_index = self.encode(" Negative")[0]
        self._prepare(pos_dir,1,self.pos_index)
        self._prepare(neg_dir,0,self.neg_index)

    def _prepare(self, path: str, label:int, label_idx:int):
        count = 0
        for fname in os.listdir(path):
            count += 1
            self.data.append([os.path.join(path,fname),label,label_idx])
        self.summary_stats[label] = count

    def get_pos_neg_indices(self):
        return {"positive": self.pos_index, "negative": self.neg_index}

    def __len__(self):
        """
        Returns the number of examples in the train/test set as specified while initializing
        """
        return len(self.data)

    def encode(self, s: str):
        return self.enc.encode(s, allowed_special={"<|endoftext|>"})

    def __getitem__(self, idx: int):
        fpath, label, label_idx = self.data[idx]

        review = open(fpath).read().replace("<br />","")
        review = f"{review}"
        review_ids_orig = self.encode(review)
        review_ids = []
        orig_review_max_len = self.max_length - len(self.prompt_prefix_ids) - len(self.prompt_suffix_ids)
        review_ids.extend(self.prompt_prefix_ids)
        review_ids.extend(review_ids_orig[orig_review_max_len-1:])
        review_ids.extend(self.prompt_suffix_ids)
        review_ids = torch.tensor(review_ids)
        attention_mask = torch.ones(review_ids.size(),dtype=torch.bool)
        return {
            "input_ids": review_ids,
            "length": len(review_ids_orig),
            "attention_mask": attention_mask,
            "label": label,
            "label_idx": label_idx,
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