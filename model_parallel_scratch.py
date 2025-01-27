import os
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Module
from torch.nn.functional import relu
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.DEBUG,  # Set the log level here
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
)

class ModelPart1(Module):
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.linear1 = torch.nn.Linear(10, 50)

    def forward(self, x):
        return relu(self.linear1(x))

class ModelPart2(Module):
    def __init__(self):
        super(ModelPart2, self).__init__()
        self.linear2 = torch.nn.Linear(50, 1)

    def forward(self, x):
        return self.linear2(x)

def train_model_parallel(rank):
    batch_size = 32
    dataset = TensorDataset(torch.randn(32 * 100, 10), torch.randn(32 * 100, 1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ModelPart1() if rank == 0 else ModelPart2()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    batch = 0
    for x, y in dataloader:
        if rank == 0:
            logging.info(f"Batch {batch}:")
            activations = model(x)
            dist.send(activations, dst=1)
            logging.debug(f"Rank 0 sent layer 1 activations to rank 1")
            grads_part1 = torch.empty_like(activations)
            dist.recv(grads_part1, src=1)
            logging.debug(f"Rank 0 received partial grads from rank 1")
            activations.backward(grads_part1)
            optimizer.step()
            optimizer.zero_grad()
        elif rank == 1:
            activations_part1 = torch.empty(batch_size, 50, requires_grad=True)
            dist.recv(activations_part1, src=0)
            logging.debug(f"Rank 1 received activations from rank 0")
            outputs_part2 = model(activations_part1)
            loss = loss_fn(outputs_part2, y)
            logging.info(f"Loss: {loss}")
            loss.backward()
            dist.send(activations_part1.grad, dst=0)
            logging.debug(f"Rank 1 sent partial grads to rank 0")
            optimizer.step()
            optimizer.zero_grad()
        batch += 1

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

if __name__ == "__main__":
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    logging.info(f"Rank {rank} is ready")
    train_model_parallel(rank)
    dist.destroy_process_group()