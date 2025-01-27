import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the log level here
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
)


class MockDataset(Dataset):
    def __init__(self, features, N):
        self.data = torch.randn(N, features)
        self.labels = torch.randint(0, 2, (N,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class TinyAllReduceModel(torch.nn.Module):

    def __init__(self, input_dim):
        super(TinyAllReduceModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 2)
        self._register_hooks()


    def _register_hooks(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(self._all_reduce_hook(name))

    @staticmethod
    def _all_reduce_hook(name):
        def _inner_all_reduce_hook(grad):
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            logging.debug(f"Rank {rank} finished all-reducing gradients for {name}")
            return grad / dist.get_world_size()
        return _inner_all_reduce_hook

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
        
def train(model):
    dataset = MockDataset(50, 10000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for inputs, labels in dataloader:
            loss = loss_func(model(inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if rank == 0:
                logging.info(f"Loss: {loss}")
                
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

if __name__ == "__main__":
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    model = TinyAllReduceModel(50)
    logging.info(f"Rank {rank} is ready")
    train(model)
    dist.destroy_process_group()