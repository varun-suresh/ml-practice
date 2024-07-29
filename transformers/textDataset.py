import random
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Dataset for the char level text dataset.
    """
    def __init__(self, split, fname, block_length=64, train_split=0.9):
        assert split in ["train", "test"]
        self.split = split
        self.train_split = train_split
        with open(fname, "r") as f:
            self.text = f.read()
        self.block_length = block_length
        self._prepare()
        self.vocab_length = len(self.stoi)

    def __len__(self):
        return len(self.text)

    def _prepare(self):
        s = sorted(list(set(self.text)))
        self.stoi = {ch:i for i,ch in enumerate(s)}
        self.itos = {i:ch for i,ch in enumerate(s)}

    def encode(self,text):
        return torch.Tensor([self.stoi[ch] for ch in text]).to(torch.int64)
    
    def decode(self, vec):
        return "".join([self.itos[i.item()] for i in vec])


    def __getitem__(self, idx):
        if self.split == "train":
            ix = random.randint(0, int(self.train_split*len(self) - self.block_length - 2)) 
        else:
            ix = random.randint((int(self.train_split*len(self) - self.block_length - 2), self.block_length - 2))
        x = self.encode(self.text[ix : ix + self.block_length])
        y = self.encode(self.text[ix+1 : ix + self.block_length + 1])
        return x,y

    def summary(self):
        """
        Print the summary of the dataset:
        Training data tokens: 
        Test data tokens:
        Vocabulary size:
        """
        print(f"Training Data tokens: {int(self.train_split * len(self))}")
        print(f"Test Data tokens: {int((1-self.train_split)*len(self))}")
        print(f"Vocabulary Size: {self.vocab_length}")

    def count_epochs(self, batch_size, iters):
        return (self.block_length*batch_size * iters)/int(self.train_split * len(self))

if __name__ == "__main__":
    fname = "data/shakespeare.txt"
    sd = TextDataset("train",fname)
    # for x, y in sd:
        # print(sd.decode(x))
        # print("----")
        # print(sd.decode(y))
        # break
    sd.summary()
    print(f"No of epochs: {sd.count_epochs(100,2000)}")