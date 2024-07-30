import torch
from torch.utils.data import DataLoader
from transformer import Transformer
from trainer import Trainer
from sortDataset import SortDataset
from textDataset import TextDataset
from dataclasses import dataclass
from utils import timeit
from transformer_config import TransformerConfig

@dataclass
class Config:
    max_iters = 2000
    batch_size = 400

def train_transformer(dataset, transformer_config, path=None):
    """
    Trains the model and saves it to the specified path
    """
    model = Transformer(transformer_config)
    config = Config
    trainer = Trainer(config, model,dataset)
    trainer.run(path)

@timeit
def eval_transformer(transformer_config,path):
    """
    Loads the transformer from the specified path and runs eval
    """
    model = Transformer(transformer_config)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device=device)
    test_dataset = SortDataset('test')
    test_loader = DataLoader(test_dataset,batch_size=64) 
    model.eval()
    n = 6
    correct = 0
    print(f"Model is on device: {next(model.parameters()).device}")
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        inp = x[:,:n]
        target = y[:,n-1:]
        with torch.no_grad():
            predicted = model.generate(inp, n)
        equal_rows = torch.all(target == predicted[:,n:], dim=1)
        correct += torch.sum(equal_rows).item()
        # break
        
    print(f"No of times correct: {correct}/{len(test_dataset)}")
    print(f"No of times incorrect: {len(test_dataset) - correct}/{len(test_dataset)}")


def train_sortDataset():
    path = "sorting_model.pk"
    transformer_config = TransformerConfig(d_model=512, d_ff=2048,d_block=11,d_vocab=3,n_decoders=6)
    train_transformer(SortDataset("train"), transformer_config,path)
    # eval_transformer(transformer_config,path)

    # model = Transformer()
    # checkpoint = torch.load(path)
    # print(checkpoint['iterations'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()

    # # n = train_dataset.length # naugy direct access shrug
    # n = 6
    # inp = torch.tensor([[0, 0, 2, 2, 1, 1]], dtype=torch.long)
    # assert inp[0].nelement() == n
    # with torch.no_grad():
    #     cat = model.generate(inp, n)
    # sol = torch.sort(inp[0])[0]
    # sol_candidate = cat[:, n:]
    # print('input sequence  :', inp.tolist())
    # print('predicted sorted:', sol_candidate.tolist())
    # print('gt sort         :', sol.tolist())
    # print('matches         :', bool((sol == sol_candidate).all()))


def train_text(model_path, text_path):
    dataset = TextDataset("train", text_path)
    transformer_config = TransformerConfig(d_model=512, 
                                           d_ff=2048, 
                                           d_block=dataset.block_length,
                                           d_vocab=dataset.vocab_length)
    train_transformer(dataset,transformer_config,model_path)

def generate_text(model_path, text_path, start_text, max_new_tokens):
    dataset = TextDataset("test", text_path)
    transformer_config = TransformerConfig(d_model=512,
                                           d_ff=2048,
                                           d_block=dataset.block_length,
                                           d_vocab=dataset.vocab_length)
    model = Transformer(transformer_config)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)
    model.eval()
    inp = dataset.encode(start_text)
    with torch.no_grad():
        predicted = model.generate(inp.view(1,-1).to(device),max_new_tokens)
    print(dataset.decode(predicted[0]))
 
if __name__ == "__main__":
    # train_sortDataset()
    model_path = "char_level_shakespeare.pk"
    text_path = "data/shakespeare.txt"
    train_text(model_path, text_path)
    generate_text(model_path, text_path,"The gods be good unto us", 1000)
