import random
from typing import List, Tuple, Union

import torch


class SimpleReplayBuffer:
    def __init__(self, num_per_task: int = 0, device: Union[torch.device, str] = "cpu"):
        self.num_per_task = num_per_task
        self.device = torch.device(device)
        self.storage: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def __len__(self):
        return sum(inputs.size(0) for inputs, _ in self.storage)

    @torch.no_grad()
    def add_examples_from_loader(self, data_loader: torch.utils.data.DataLoader, task_id: int):
        if self.num_per_task <= 0:
            return

        if task_id < len(self.storage):
            return

        inputs_buffer, targets_buffer = [], []
        for inputs, targets in data_loader:
            inputs_buffer.append(inputs.cpu())
            targets_buffer.append(targets.cpu())
            if len(torch.cat(inputs_buffer)) >= self.num_per_task:
                break

        if len(inputs_buffer) == 0:
            return

        inputs_cat = torch.cat(inputs_buffer)[: self.num_per_task]
        targets_cat = torch.cat(targets_buffer)[: self.num_per_task]
        self.storage.append((inputs_cat, targets_cat))

    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Replay buffer is empty – cannot sample.")

        inputs_all, targets_all = [], []
        for inputs, targets in self.storage:
            inputs_all.append(inputs)
            targets_all.append(targets)
        inputs_all = torch.cat(inputs_all)
        targets_all = torch.cat(targets_all)

        idx = torch.randperm(inputs_all.size(0))[:n_samples]
        return inputs_all[idx].to(self.device), targets_all[idx].to(self.device)
