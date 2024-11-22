"""
Evaluate fine-tuned GPT-2 on IMDb movie reviews
"""

from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from sentiment_classification.reviewsDataset import reviewsDataset
from gpt_utils import dynamic_padding
from gpt_config import GPTConfig
from sentiment_classification.eval_config import EvalConfig
from gpt import GPT

class Eval:
    def __init__(self,test_set: reviewsDataset,eval_config: EvalConfig, model_config: GPTConfig):
        self.test_set = test_set
        self.eval_config = eval_config
        self.model_config = model_config
        self.load_model()
    
    def load_model(self):
        self.model = GPT.from_pretrained(config=self.model_config)
        if self.model_config.load_from_checkpoint:
            ckpt = torch.load(self.model_config.checkpoint_path,map_location=self.eval_config.device)
            self.model.load_state_dict(ckpt["model"])
        self.model.to(self.eval_config.device)
        self.model.eval()

    def evaluate(self):
        if self.eval_config.subset:
            subset_range = torch.arange(0,len(self.test_set),self.eval_config.interval)
            dl = DataLoader(torch.utils.data.Subset(self.test_set,subset_range),batch_size=self.eval_config.batch_size,collate_fn=dynamic_padding)
        else:
            dl = DataLoader(self.test_set,batch_size=self.eval_config.batch_size,collate_fn=dynamic_padding)

        results_file = open(self.eval_config.results_path,"w")
        results_file.write("filename,length,label,prediction,logit_pos,logit_neg\n")
        for batch in tqdm(dl):
            with torch.no_grad():
                logits, _ = self.model(batch["input_ids"].to(self.eval_config.device),batch["attention_masks"].to(self.eval_config.device))
                sentiment_idx = self.test_set.get_pos_neg_indices()
                for i, fname in enumerate(batch["fpaths"]):
                    pos = logits[i,0,sentiment_idx["positive"]]
                    neg = logits[i,0,sentiment_idx["negative"]]
                    if pos > neg:
                        prediction = 1
                    else:
                        prediction = 0
    
                    results_file.write(f"{fname},{batch['lengths'][i]},{batch['labels'][i]},{prediction},{pos},{neg}\n")               

