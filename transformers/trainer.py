import torch
from torch.utils.data import DataLoader
import time

class Trainer:
    def __init__(self,config,model, train_dataset):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.iter_num = 0
        self.iter_time = 0
        self.iter_dt = 0
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def run(self,path=None):
        model, config = self.model, self.config
        model = model.to(self.device)
        self.optimizer = model.configure_optimizers(self.config)
        train_loader = DataLoader(self.train_dataset, shuffle=True,batch_size=config.batch_size)
        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x,y = batch
            # print(f"Training Data: x: {x}, y: {y}")
            logits, self.loss = model(x,y)

            model.zero_grad(set_to_none=True)
            self.loss.backward()
            self.optimizer.step()
            self.iter_dt = time.time() - self.iter_time
            self.iter_time = time.time()
            if self.iter_num % 100 == 0:
                print(f"Iteration: {self.iter_num}, Loss: {self.loss}, Time per iter: {self.iter_dt}")
            self.iter_num += 1
            if self.config.max_iters is not None and self.iter_num >= self.config.max_iters:
                break
        if path:
            torch.save({
                'iterations': self.iter_num,
                'model_state_dict': model.state_dict(),
                'loss': self.loss,
            }, path)
