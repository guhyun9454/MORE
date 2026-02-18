import argparse
import datetime
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from continual_datasets.build_incremental_scenario import build_continual_dataloader
from continual_datasets.dataset_utils import get_ood_dataset, set_data_config
from networks.my_vit_hat import deit_small_patch16_224, vit_base_patch16_224
from utils.sgd_hat import SGD_hat


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy (percent)."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / target.size(0)))
        return res


def tensor_total_bytes(t: torch.Tensor) -> int:
    # Total Bytes = tensor.element_size() * tensor.nelement()
    return int(t.element_size() * t.nelement())


def save_accuracy_heatmap(acc_matrix: np.ndarray, task_id: int, args) -> str:
    """
    Save an upper-triangular accuracy heatmap (like oodvil code.md).
    """
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    mat = np.ma.masked_invalid(acc_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=0, vmax=100, interpolation="nearest")
    ax.set_title(f"Accuracy Heatmap (till task {task_id+1})")
    ax.set_xlabel("Learned task")
    ax.set_ylabel("Eval task")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = os.path.join(args.output_dir, f"accuracy_heatmap_task_{task_id+1}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _random_subset(dataset: torch.utils.data.Dataset, n_samples: int, seed: int) -> torch.utils.data.Dataset:
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    n_total = len(dataset)
    if n_samples >= n_total:
        return dataset
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_total, size=n_samples, replace=False).tolist()
    return torch.utils.data.Subset(dataset, idx)


def load_more_deit_pretrain_if_available(net: nn.Module, ckpt_path: str, require: bool = False) -> bool:
    """
    Load the MORE-provided DeiT pre-trained checkpoint (README: ./deit_pretrained/best_checkpoint.pth).
    Returns True if loaded, False if not found (unless require=True).
    """
    ckpt_path = str(ckpt_path)
    if ckpt_path and os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        pretrain = checkpoint.get("model", checkpoint)
        target = net.state_dict()
        transfer = {k: v for k, v in pretrain.items() if (k in target) and ("head" not in k)}
        target.update(transfer)
        net.load_state_dict(target)
        return True

    if require:
        raise FileNotFoundError(
            f"Cannot find MORE pre-trained checkpoint at '{ckpt_path}'. "
            "Download it per README and place it at ./deit_pretrained/best_checkpoint.pth "
            "(or pass --more_pretrain_path)."
        )
    return False


@torch.no_grad()
def _compute_more_confidence_scores(model: nn.Module, data_loader, device: torch.device) -> np.ndarray:
    """
    MORE OOD score in the paper (Algorithm 2, step 6):
    s(x) = max_{k<=t} p(Y_k|x,k) * s_k(x)

    This model returns log-scores over global classes, so we can use max-log-score
    as a monotonic confidence score for AUROC/FPR@TPR95.
    """
    model.eval()
    scores: List[torch.Tensor] = []
    for batch_idx, (inputs, _targets) in enumerate(data_loader):
        inputs = inputs.to(device, non_blocking=True)
        out = model(inputs)  # [B, C] log-scores
        conf = out.max(dim=1).values.detach().cpu()
        scores.append(conf)
    if len(scores) == 0:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(scores, dim=0).to(dtype=torch.float32).numpy()


def _auroc_and_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute AUROC and FPR@TPR95 without sklearn.

    Convention:
    - ID is positive (label=1) and should have higher score than OOD (label=0).
    """
    id_scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if id_scores.size == 0 or ood_scores.size == 0:
        raise ValueError("Empty score array for AUROC computation.")

    scores = np.concatenate([id_scores, ood_scores], axis=0)
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)], axis=0)

    # sort by descending score (stable sort for tie handling)
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]

    P = float(labels.sum())
    N = float(labels.size - P)
    if P <= 0 or N <= 0:
        raise ValueError("Invalid positive/negative counts for AUROC computation.")

    tps = np.cumsum(labels)
    fps = np.cumsum(1.0 - labels)

    tpr = tps / P
    fpr = fps / N

    # add origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auroc = float(np.trapz(tpr, fpr))
    idx = int(np.searchsorted(tpr, 0.95, side="left"))
    idx = min(max(idx, 0), int(fpr.size - 1))
    fpr95 = float(fpr[idx])
    return auroc, fpr95


def evaluate_ood_detection(
    model: nn.Module,
    id_dataset: torch.utils.data.Dataset,
    ood_dataset: torch.utils.data.Dataset,
    device: torch.device,
    args,
    task_id: int,
    tag: str,
    auc_history: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Evaluate OOD detection using MORE confidence score (max output with MD coefficient).
    """
    id_size, ood_size = int(len(id_dataset)), int(len(ood_dataset))
    use_n = min(id_size, ood_size)
    if int(getattr(args, "ood_eval_samples", 0)) > 0:
        use_n = min(use_n, int(args.ood_eval_samples))
    if bool(getattr(args, "develop", False)):
        use_n = min(use_n, 1000)
    if use_n <= 0:
        raise ValueError("Cannot run OOD eval with non-positive sample count.")

    id_ds = _random_subset(id_dataset, use_n, seed=int(args.seed) + 12345) if id_size > use_n else id_dataset
    ood_ds = _random_subset(ood_dataset, use_n, seed=int(args.seed) + 54321) if ood_size > use_n else ood_dataset

    id_loader = torch.utils.data.DataLoader(
        id_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True
    )
    ood_loader = torch.utils.data.DataLoader(
        ood_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True
    )

    id_scores = _compute_more_confidence_scores(model, id_loader, device)
    ood_scores = _compute_more_confidence_scores(model, ood_loader, device)
    auroc, fpr95 = _auroc_and_fpr95(id_scores, ood_scores)

    if auc_history is not None:
        auc_history.append(float(auroc))
        iauc = float(np.mean(np.array(auc_history, dtype=np.float64)))
    else:
        iauc = float("nan")

    print(
        f"[{tag}] (TASK {task_id+1}) AUROC {auroc * 100:.2f}% | IAUC {iauc * 100:.2f}% | "
        f"FPR@TPR95 {fpr95 * 100:.2f}% | N={use_n}"
    )

    if args.wandb:
        import wandb

        payload = {
            f"{tag}_AUROC (↑)": auroc * 100.0,
            f"{tag}_FPR@TPR95 (↓)": fpr95 * 100.0,
            "TASK": task_id,
            f"{tag}_N": use_n,
        }
        if not math.isnan(iauc):
            payload[f"{tag}_IAUC (↑)"] = iauc * 100.0
        wandb.log(payload)

    return {"auroc": float(auroc), "fpr_at_tpr95": float(fpr95), "n": float(use_n), "iauc": float(iauc)}


class ByteReplayBuffer:
    """
    Replay buffer with capacity specified in *bytes*.
    Stored tensors are kept on CPU; sampling moves them to the requested device.

    NOTE (VIL compatibility):
    - We additionally store the originating task-id per sample so back-update can
      draw IND samples for a specific previous task even when global class labels
      overlap across tasks (common in VIL scenarios).
    """

    def __init__(self, capacity_bytes: int, seed: int = 0, store_dtype: str = "uint8"):
        self.capacity_bytes = int(capacity_bytes)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.store_dtype = str(store_dtype).lower()
        if self.store_dtype not in {"uint8", "float32"}:
            raise ValueError("store_dtype must be one of: {'uint8', 'float32'}")

        self._bytes_per_sample: Optional[int] = None
        # key = (task_id, global_class)
        self._x_by_key: Dict[Tuple[int, int], torch.Tensor] = {}
        self._y_by_key: Dict[Tuple[int, int], torch.Tensor] = {}

        self._x_all: Optional[torch.Tensor] = None
        self._y_all: Optional[torch.Tensor] = None
        self._t_all: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        if self._y_all is None:
            return 0
        return int(self._y_all.numel())

    def bytes_used(self) -> int:
        if self._x_all is None or self._y_all is None or self._t_all is None:
            return 0
        return tensor_total_bytes(self._x_all) + tensor_total_bytes(self._y_all) + tensor_total_bytes(self._t_all)

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        # Store images more byte-efficiently when capacity is specified in bytes.
        # ToTensor() yields float in [0,1] with 8-bit granularity, so uint8 round-trip is effectively lossless.
        if self.store_dtype == "uint8":
            return (x * 255.0).round().clamp(0, 255).to(dtype=torch.uint8)
        return x.to(dtype=torch.float32)

    def _decode_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.store_dtype == "uint8":
            return x.to(dtype=torch.float32) / 255.0
        return x

    def _rebuild_flat_cache(self):
        if len(self._x_by_key) == 0:
            self._x_all = None
            self._y_all = None
            self._t_all = None
            return
        xs, ys = [], []
        ts: List[torch.Tensor] = []
        for task_id, y in sorted(self._x_by_key.keys()):
            xk = self._x_by_key[(task_id, y)]
            yk = self._y_by_key[(task_id, y)]
            xs.append(xk)
            ys.append(yk)
            ts.append(torch.full((xk.size(0),), fill_value=int(task_id), dtype=torch.int64))
        self._x_all = torch.cat(xs, dim=0).contiguous()
        self._y_all = torch.cat(ys, dim=0).contiguous()
        self._t_all = torch.cat(ts, dim=0).contiguous()

    def _infer_bytes_per_sample(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        if self._bytes_per_sample is not None:
            return
        # store per-sample bytes using the requested formula
        self._bytes_per_sample = tensor_total_bytes(x) + tensor_total_bytes(y) + tensor_total_bytes(t)

    def _max_samples_total(self) -> int:
        if self.capacity_bytes <= 0 or self._bytes_per_sample is None or self._bytes_per_sample <= 0:
            return 0
        return int(self.capacity_bytes // self._bytes_per_sample)

    @torch.no_grad()
    def update_from_loader(self, data_loader: torch.utils.data.DataLoader, task_id: int, new_classes: List[int]):
        """
        Merge new samples (from current task) into buffer, then downsample per-class to fit the
        total byte budget while keeping per-(task,class) counts equal.
        """
        if self.capacity_bytes <= 0:
            return

        task_id = int(task_id)

        # infer bytes/sample from the first available sample
        if self._bytes_per_sample is None:
            for x, y in data_loader:
                if x.numel() == 0:
                    continue
                x0 = self._encode_x(x[0].detach().cpu())
                y0 = y[0].detach().cpu()
                t0 = torch.tensor(task_id, dtype=torch.int64)
                self._infer_bytes_per_sample(x0, y0, t0)
                break

        max_total = self._max_samples_total()
        if max_total <= 0:
            # cannot store even 1 sample within byte budget
            self._x_by_key = {}
            self._y_by_key = {}
            self._rebuild_flat_cache()
            return

        existing_keys = set(self._x_by_key.keys())
        new_keys = [(task_id, int(c)) for c in new_classes]
        seen_after = sorted(existing_keys.union(set(new_keys)))
        if len(seen_after) == 0:
            return

        per_key_quota = int(max_total // len(seen_after))
        if per_key_quota <= 0:
            # not enough capacity to allocate >=1 per seen (task,class)
            self._x_by_key = {}
            self._y_by_key = {}
            self._rebuild_flat_cache()
            return

        # collect candidates from current loader (early-stop when we have enough per class)
        new_class_set = set(int(c) for c in new_classes)
        need = {(task_id, int(c)): per_key_quota for c in new_classes}
        collected_x: Dict[Tuple[int, int], List[torch.Tensor]] = {(task_id, int(c)): [] for c in new_classes}
        collected_y: Dict[Tuple[int, int], List[torch.Tensor]] = {(task_id, int(c)): [] for c in new_classes}

        for x, y in data_loader:
            x_cpu = self._encode_x(x.detach().cpu())
            y_cpu = y.detach().cpu()
            for i in range(x_cpu.size(0)):
                cls = int(y_cpu[i].item())
                if cls in new_class_set:
                    key = (task_id, cls)
                    if key in need and need[key] > 0:
                        collected_x[key].append(x_cpu[i : i + 1])  # keep batch dim
                        collected_y[key].append(y_cpu[i : i + 1])
                        need[key] -= 1

            if all(v <= 0 for v in need.values()):
                break

        # merge candidates
        for cls in new_classes:
            cls = int(cls)
            key = (task_id, cls)
            if len(collected_x.get(key, [])) == 0:
                continue
            x_new = torch.cat(collected_x[key], dim=0)
            y_new = torch.cat(collected_y[key], dim=0)

            if key in self._x_by_key:
                self._x_by_key[key] = torch.cat([self._x_by_key[key], x_new], dim=0)
                self._y_by_key[key] = torch.cat([self._y_by_key[key], y_new], dim=0)
            else:
                self._x_by_key[key] = x_new
                self._y_by_key[key] = y_new

        # downsample all seen (task,class) to per_key_quota
        for key in list(self._x_by_key.keys()):
            x_cls = self._x_by_key[key]
            if x_cls.size(0) > per_key_quota:
                idx = torch.from_numpy(self.rng.permutation(x_cls.size(0))[:per_key_quota]).long()
                self._x_by_key[key] = x_cls[idx].contiguous()
                self._y_by_key[key] = self._y_by_key[key][idx].contiguous()

        self._rebuild_flat_cache()

        # final safety (should already hold)
        if self.bytes_used() > self.capacity_bytes:
            # fallback: global truncate (rare, e.g., if shapes vary)
            if self._x_all is not None and self._y_all is not None and self._t_all is not None:
                max_keep = max_total
                self._x_all = self._x_all[:max_keep].contiguous()
                self._y_all = self._y_all[:max_keep].contiguous()
                self._t_all = self._t_all[:max_keep].contiguous()

    def sample(self, n_samples: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Replay buffer is empty – cannot sample.")
        n = min(int(n_samples), len(self))
        idx = torch.randperm(len(self))[:n]
        assert self._x_all is not None and self._y_all is not None and self._t_all is not None
        x = self._decode_x(self._x_all[idx]).to(device)
        return x, self._y_all[idx].to(device), self._t_all[idx].to(device)

    def sample_in_task(
        self, task_id: int, n_samples: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Replay buffer is empty – cannot sample.")
        assert self._x_all is not None and self._y_all is not None and self._t_all is not None

        task_id = int(task_id)
        mask = self._t_all == task_id
        idx_all = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx_all.numel() == 0:
            raise ValueError("No samples found for the requested task.")

        n = min(int(n_samples), int(idx_all.numel()))
        sel = idx_all[torch.randperm(idx_all.numel())[:n]]
        x_sel = self._decode_x(self._x_all[sel])
        y_sel = self._y_all[sel]
        t_sel = self._t_all[sel]
        if device is not None:
            x_sel = x_sel.to(device)
            y_sel = y_sel.to(device)
            t_sel = t_sel.to(device)
        return x_sel, y_sel, t_sel

    def sample_not_in_task(
        self, task_id: int, n_samples: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Replay buffer is empty – cannot sample.")
        assert self._x_all is not None and self._y_all is not None and self._t_all is not None

        task_id = int(task_id)
        mask = self._t_all != task_id
        idx_all = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx_all.numel() == 0:
            raise ValueError("No samples found outside the requested task.")

        n = min(int(n_samples), int(idx_all.numel()))
        sel = idx_all[torch.randperm(idx_all.numel())[:n]]
        x_sel = self._decode_x(self._x_all[sel])
        y_sel = self._y_all[sel]
        t_sel = self._t_all[sel]
        if device is not None:
            x_sel = x_sel.to(device)
            y_sel = y_sel.to(device)
            t_sel = t_sel.to(device)
        return x_sel, y_sel, t_sel

    def sample_in_classes(
        self, class_set: set, n_samples: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Replay buffer is empty – cannot sample.")
        assert self._x_all is not None and self._y_all is not None

        if len(class_set) == 0:
            raise ValueError("class_set is empty.")

        y = self._y_all
        mask = torch.zeros_like(y, dtype=torch.bool)
        for c in class_set:
            mask |= (y == int(c))
        idx_all = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx_all.numel() == 0:
            raise ValueError("No samples found for the requested classes.")

        n = min(int(n_samples), int(idx_all.numel()))
        sel = idx_all[torch.randperm(idx_all.numel())[:n]]
        x_sel = self._decode_x(self._x_all[sel])
        y_sel = self._y_all[sel]
        if device is not None:
            x_sel = x_sel.to(device)
            y_sel = y_sel.to(device)
        return x_sel, y_sel

    def sample_not_in_classes(
        self, class_set: set, n_samples: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Replay buffer is empty – cannot sample.")
        assert self._x_all is not None and self._y_all is not None

        y = self._y_all
        mask = torch.ones_like(y, dtype=torch.bool)
        for c in class_set:
            mask &= (y != int(c))
        idx_all = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx_all.numel() == 0:
            raise ValueError("No samples found outside the requested classes.")

        n = min(int(n_samples), int(idx_all.numel()))
        sel = idx_all[torch.randperm(idx_all.numel())[:n]]
        x_sel = self._decode_x(self._x_all[sel])
        y_sel = self._y_all[sel]
        if device is not None:
            x_sel = x_sel.to(device)
            y_sel = y_sel.to(device)
        return x_sel, y_sel


@dataclass
class MoreStats:
    means: torch.Tensor  # [K, D] (local class order)
    cov_inv: torch.Tensor  # [D, D]


class MoreOODVILWrapper(nn.Module):
    """
    Task-free inference wrapper for MORE:
    - Multi-head (task-specific) classifiers with an extra OOD class
    - Mahalanobis-based coefficient per task
    - Aggregate to global class logits (handles overlapping classes across tasks via per-class max)
    """

    def __init__(
        self,
        net: nn.Module,
        class_mask: List[List[int]],
        num_classes: int,
        num_cls_per_task: int,
        smax: float = 500.0,
        md_c: float = 20.0,
        infer_T: float = 2.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.net = net
        self.class_mask = class_mask
        self.num_classes = int(num_classes)
        self.num_cls_per_task = int(num_cls_per_task)
        self.smax = float(smax)
        self.md_c = float(md_c)
        self.infer_T = float(infer_T)
        self.eps = float(eps)

        self.learned_tasks: int = 0
        self.stats: List[Optional[MoreStats]] = []

    def set_learned_tasks(self, n: int):
        self.learned_tasks = int(n)

    def set_task_stats(self, task_id: int, means: torch.Tensor, cov_inv: torch.Tensor):
        while len(self.stats) <= task_id:
            self.stats.append(None)
        self.stats[task_id] = MoreStats(means=means.detach().cpu(), cov_inv=cov_inv.detach().cpu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned_tasks <= 0:
            raise RuntimeError("No learned tasks yet.")

        device = x.device
        B = x.size(0)
        global_scores = torch.zeros((B, self.num_classes), device=device, dtype=torch.float32)

        for t in range(self.learned_tasks):
            features, _ = self.net.forward_features(t, x, s=self.smax)
            logits_t = self.net.forward_classifier(t, features)
            probs_t = F.softmax(logits_t / self.infer_T, dim=1)[:, : self.num_cls_per_task]

            st = None
            if t < len(self.stats) and self.stats[t] is not None:
                means = self.stats[t].means.to(device=device, dtype=features.dtype)  # [K, D]
                cov_inv = self.stats[t].cov_inv.to(device=device, dtype=features.dtype)  # [D, D]

                # Mahalanobis distance for all classes at once: [B, K]
                delta = features[:, None, :] - means[None, :, :]
                maha = torch.einsum("bkd,dd,bkd->bk", delta, cov_inv, delta)
                dist = torch.sqrt(torch.clamp(maha, min=self.eps))
                scores = self.md_c / dist
                st = scores.max(dim=1, keepdim=True).values  # [B, 1]

            if st is not None:
                probs_t = probs_t * st

            # scatter to global classes (amax for overlaps)
            for local_i, global_cls in enumerate(self.class_mask[t]):
                global_scores[:, int(global_cls)] = torch.maximum(global_scores[:, int(global_cls)], probs_t[:, local_i])

        return torch.log(global_scores + self.eps)


def hat_reg(p_mask: Optional[Dict[str, torch.Tensor]], masks: List[torch.Tensor], lamb0: float, lamb1: float) -> torch.Tensor:
    """
    Sparsity regularization from HAT used in MORE.
    """
    reg = 0.0
    count = 0.0
    if p_mask is not None:
        for m, mp in zip(masks, p_mask.values()):
            aux = 1.0 - mp.to(m.device)
            reg += (m * aux).sum()
            count += aux.sum()
        reg = reg / (count + 1e-12)
        return lamb1 * reg
    else:
        for m in masks:
            reg += m.sum()
            count += float(np.prod(list(m.size())))
        reg = reg / (count + 1e-12)
        return lamb0 * reg


@torch.no_grad()
def cum_mask(net: nn.Module, t: int, p_mask: Optional[Dict[str, torch.Tensor]], smax: float, device: torch.device):
    task_id = torch.tensor([t]).to(device)
    mask: Dict[str, torch.Tensor] = {}
    for n, _ in net.named_parameters():
        names = n.split(".")
        checker = [i for i in ["ec0", "ec1", "ec2"] if i in names]
        if checker and "adapter" in n:
            # e.g., blocks.0.adapter1.ec1.0
            blk = int(names[1])
            adapter_name = names[2]  # adapter1 / adapter2
            adapter = net.blocks[blk].__getattr__(adapter_name)
            gc1, gc2 = adapter.mask(task_id, s=smax)
            if checker[0] == "ec1":
                key = ".".join(n.split(".")[:-1])
                mask[key] = gc1.detach().cpu()
                mask[key].requires_grad = False
            elif checker[0] == "ec2":
                key = ".".join(n.split(".")[:-1])
                mask[key] = gc2.detach().cpu()
                mask[key].requires_grad = False

    if p_mask is None:
        p_mask = {k: v for k, v in mask.items()}
    else:
        for k, v in mask.items():
            p_mask[k] = torch.max(p_mask[k], v)
    return p_mask


@torch.no_grad()
def freeze_mask(net: nn.Module, p_mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mask_back: Dict[str, torch.Tensor] = {}
    for n, p in net.named_parameters():
        names = n.split(".")
        if "adapter" in n:
            # blocks.<i>.<adapter1|adapter2>.<fc1|fc2>.<weight|bias>
            if "fc1.weight" in n:
                key = ".".join(names[:-2]) + ".ec1"
                mask_back[n] = 1 - p_mask[key].data.view(-1, 1).expand_as(p).to(p.device)
            elif "fc1.bias" in n:
                key = ".".join(names[:-2]) + ".ec1"
                mask_back[n] = 1 - p_mask[key].data.view(-1).to(p.device)
            elif "fc2.weight" in n:
                key1 = ".".join(names[:-2]) + ".ec1"
                key2 = ".".join(names[:-2]) + ".ec2"
                post = p_mask[key2].data.view(-1, 1).expand_as(p).to(p.device)
                pre = p_mask[key1].data.view(1, -1).expand_as(p).to(p.device)
                mask_back[n] = 1 - torch.min(post, pre)
            elif "fc2.bias" in n:
                key = ".".join(names[:-2]) + ".ec2"
                mask_back[n] = 1 - p_mask[key].data.view(-1).to(p.device)
    return mask_back


def compensation(net: nn.Module, thres_cosh: float, s: float, smax: float):
    # Equation before Eq. (4) in HAT (used in MORE codebase)
    for n, p in net.named_parameters():
        if "ec" in n and p.grad is not None:
            num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
            den = torch.cosh(p.data) + 1
            p.grad *= (smax / s) * (num / den)


@torch.no_grad()
def compensation_clamp(net: nn.Module, thres_emb: float):
    for n, p in net.named_parameters():
        if "ec" in n:
            p.data.copy_(torch.clamp(p.data, -thres_emb, thres_emb))


def prepare_hat(net: nn.Module, mask_back: Optional[Dict[str, torch.Tensor]]):
    for n, p in net.named_parameters():
        p.grad = None
        if mask_back is not None and n in mask_back:
            p.hat = mask_back[n]
        else:
            p.hat = None


def set_requires_grad_for_more(net: nn.Module, freeze_head: bool):
    for p in net.parameters():
        p.requires_grad = False
    # enable only adapter + task norms (+ head if not frozen)
    for n, p in net.named_parameters():
        if ("adapter" in n) or ("list_norm" in n) or (("head" in n) and (not freeze_head)):
            p.requires_grad = True


@torch.no_grad()
def compute_task_stats(
    net: nn.Module,
    task_id: int,
    train_loader: torch.utils.data.DataLoader,
    class_to_local: Dict[int, int],
    num_cls_per_task: int,
    smax: float,
    device: torch.device,
    max_samples_per_class: int,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MORE inference statistics:
    - per-class mean (local order) μ^k_j
    - task covariance inverse S^{-1} (average of class covariances)
    """
    net.eval()

    # infer feature dim
    # (MyVisionTransformer uses embed_dim = 768 for vit_base_patch16_224)
    feat_dim = int(getattr(net, "embed_dim", 768))

    sums = torch.zeros((num_cls_per_task, feat_dim), dtype=torch.float64)
    counts = torch.zeros((num_cls_per_task,), dtype=torch.long)

    # store only a limited number of features per class for covariance estimation
    max_samples_per_class = int(max_samples_per_class)
    feat_store: List[List[torch.Tensor]] = [[] for _ in range(num_cls_per_task)]
    feat_store_counts = [0 for _ in range(num_cls_per_task)]

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets_cpu = targets.detach().cpu()

        features, _ = net.forward_features(task_id, inputs, s=smax)
        feats_f32 = features.detach().cpu().to(dtype=torch.float32)

        local = torch.tensor([class_to_local[int(y)] for y in targets_cpu.tolist()], dtype=torch.long)
        for cls_local in local.unique().tolist():
            cls_local = int(cls_local)
            idx = (local == cls_local)
            X = feats_f32[idx]  # [n, D] float32 on CPU
            if X.numel() == 0:
                continue

            counts[cls_local] += X.size(0)
            sums[cls_local] += X.to(dtype=torch.float64).sum(dim=0)

            if max_samples_per_class > 0 and feat_store_counts[cls_local] < max_samples_per_class:
                remaining = max_samples_per_class - feat_store_counts[cls_local]
                take = min(int(remaining), int(X.size(0)))
                if take > 0:
                    feat_store[cls_local].append(X[:take].contiguous())
                    feat_store_counts[cls_local] += take

    means = torch.zeros((num_cls_per_task, feat_dim), dtype=torch.float64)
    cov_sum = torch.zeros((feat_dim, feat_dim), dtype=torch.float64)

    valid = 0
    for k in range(num_cls_per_task):
        n = int(counts[k].item())
        if n <= 0:
            continue
        mu = sums[k] / float(n)
        means[k] = mu
        valid += 1

    if valid <= 0:
        raise RuntimeError("Cannot compute task statistics (no samples).")

    # average class covariance (estimated from a capped number of samples per class)
    cov_valid = 0
    means_f32 = means.to(dtype=torch.float32)
    for k in range(num_cls_per_task):
        if feat_store_counts[k] <= 1:
            continue
        X = torch.cat(feat_store[k], dim=0).to(dtype=torch.float64)  # [n, D]
        X = X - means[k].view(1, -1)  # center by (full) class mean
        denom = float(max(int(X.size(0)) - 1, 1))
        cov_k = X.t().matmul(X) / denom
        cov_sum += cov_k
        cov_valid += 1

    if cov_valid <= 0:
        # fallback: pooled covariance across all classes (rare if max_samples_per_class is too small)
        cov = torch.eye(feat_dim, dtype=torch.float64)
    else:
        cov = cov_sum / float(cov_valid)

    cov = cov + eps * torch.eye(feat_dim, dtype=torch.float64)
    cov_inv = torch.linalg.inv(cov)

    return means_f32, cov_inv.to(dtype=torch.float32)


@torch.no_grad()
def collect_samples_from_loader(
    data_loader: torch.utils.data.DataLoader,
    predicate,
    n_samples: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Collect up to n_samples (x, y) from a loader that satisfy predicate(global_y)->bool.
    Returns CPU tensors with batch dimension.
    """
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    n_samples = int(n_samples)
    if n_samples <= 0:
        return None, None

    for x, y in data_loader:
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        for i in range(x_cpu.size(0)):
            yi = int(y_cpu[i].item())
            if predicate(yi):
                xs.append(x_cpu[i : i + 1])
                ys.append(y_cpu[i : i + 1])
                if len(xs) >= n_samples:
                    break
        if len(xs) >= n_samples:
            break

    if len(xs) == 0:
        return None, None
    return torch.cat(xs, dim=0).contiguous(), torch.cat(ys, dim=0).contiguous()


def back_update_previous_heads(
    net: nn.Module,
    current_task_id: int,
    replay: ByteReplayBuffer,
    current_train_loader: torch.utils.data.DataLoader,
    class_mask: List[List[int]],
    task_class_to_local: List[Dict[int, int]],
    args,
    device: torch.device,
):
    """
    Back-updating phase (MORE):
    After learning task k, update previous heads so that current data is recognized as OOD.

    In VIL (possible class overlap across tasks), we treat samples whose *global class* is in
    the previous task's class set as IND; otherwise as OOD.
    """
    if not args.back_update:
        return
    if current_task_id <= 0:
        return
    if len(replay) == 0:
        return

    net.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    per_head = int(args.back_update_samples)
    bs = int(args.back_update_batch_size)

    for prev_task in range(current_task_id):
        # IND samples from replay (task-specific)
        try:
            x_ind, y_ind, _t_ind = replay.sample_in_task(prev_task, per_head, device=None)
        except Exception:
            continue

        y_ind_local = torch.tensor(
            [task_class_to_local[prev_task][int(y)] for y in y_ind.view(-1).tolist()],
            dtype=torch.long,
        )

        # OOD samples: replay samples not in this task + (if needed) samples from current task loader
        x_ood_parts: List[torch.Tensor] = []
        need_ood = int(x_ind.size(0))
        try:
            x_ood_buf, _y_ood_buf, _t_ood_buf = replay.sample_not_in_task(prev_task, need_ood, device=None)
            x_ood_parts.append(x_ood_buf)
            need_ood -= int(x_ood_buf.size(0))
        except Exception:
            pass

        if need_ood > 0:
            x_ood_cur, _y_ood_cur = collect_samples_from_loader(
                current_train_loader, predicate=lambda _yy: True, n_samples=need_ood
            )
            if x_ood_cur is not None:
                x_ood_parts.append(x_ood_cur)

        if len(x_ood_parts) == 0:
            continue
        x_ood = torch.cat(x_ood_parts, dim=0).contiguous()
        y_ood = torch.full((x_ood.size(0),), fill_value=args.num_cls_per_task, dtype=torch.long)

        x_all = torch.cat([x_ind, x_ood], dim=0).contiguous()
        y_all = torch.cat([y_ind_local, y_ood], dim=0).contiguous()

        # shuffle
        perm = torch.randperm(x_all.size(0))
        x_all = x_all[perm]
        y_all = y_all[perm]

        ds = torch.utils.data.TensorDataset(x_all, y_all)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)

        opt = torch.optim.SGD(net.head[prev_task].parameters(), lr=float(args.back_update_lr), momentum=float(args.momentum))

        for _ in range(int(args.back_update_epochs)):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                with torch.no_grad():
                    feats, _ = net.forward_features(prev_task, xb, s=float(args.smax))
                logits = net.forward_classifier(prev_task, feats)
                loss = criterion(logits, yb)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()


class TrainerOODVILMore:
    def __init__(self):
        pass

    def train_one_epoch(
        self,
        net: nn.Module,
        task_id: int,
        data_loader: torch.utils.data.DataLoader,
        optimizer: SGD_hat,
        device: torch.device,
        class_to_local: Dict[int, int],
        replay: ByteReplayBuffer,
        p_mask: Optional[Dict[str, torch.Tensor]],
        args,
    ) -> Tuple[float, float]:
        criterion = nn.CrossEntropyLoss().to(device)
        net.train()

        total_loss = 0.0
        total_loss_count = 0
        ind_correct = 0
        ind_total = 0

        num_batches = max(len(data_loader), 1)
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if args.develop and batch_idx > 20:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            local_targets = torch.tensor(
                [class_to_local[int(y)] for y in targets.detach().cpu().tolist()],
                device=device,
                dtype=torch.long,
            )

            # HAT s schedule
            s = (args.smax - 1.0 / args.smax) * (batch_idx / float(num_batches)) + (1.0 / args.smax)

            inputs_all = inputs
            labels_all = local_targets
            if args.replay_buffer_bytes > 0 and len(replay) > 0 and args.replay_batch_size > 0:
                x_bf, _y_bf_global, _t_bf = replay.sample(args.replay_batch_size, device=device)
                # MORE (Kim et al., 2022): treat all buffer samples as OOD for the current task
                y_bf_local = torch.full(
                    (x_bf.size(0),),
                    fill_value=int(args.num_cls_per_task),
                    device=device,
                    dtype=torch.long,
                )
                inputs_all = torch.cat([inputs, x_bf], dim=0)
                labels_all = torch.cat([local_targets, y_bf_local], dim=0)

            features, masks = net.forward_features(task_id, inputs_all, s=s)
            logits = net.forward_classifier(task_id, features)

            loss = criterion(logits, labels_all)
            loss = loss + hat_reg(p_mask, masks, args.lamb0, args.lamb1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            compensation(net, args.thres_cosh, s=s, smax=args.smax)
            optimizer.step(hat=(task_id > 0))
            compensation_clamp(net, args.thres_emb)

            # stats (IND only)
            with torch.no_grad():
                logits_ind = logits[: inputs.size(0), : args.num_cls_per_task]
                pred = logits_ind.argmax(dim=1)
                ind_correct += int(pred.eq(local_targets).sum().item())
                ind_total += int(local_targets.numel())

            total_loss += float(loss.item()) * int(labels_all.numel())
            total_loss_count += int(labels_all.numel())

        avg_loss = total_loss / max(total_loss_count, 1)
        avg_acc = 100.0 * float(ind_correct) / max(ind_total, 1)
        return avg_loss, avg_acc

    def evaluate_task(self, model: nn.Module, data_loader, device: torch.device, task_id: int, args) -> float:
        criterion = nn.CrossEntropyLoss().to(device)
        model.eval()
        total_acc = 0.0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if args.develop and batch_idx > 20:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                batch_size = inputs.size(0)

                total_acc += acc1.item() * batch_size
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if batch_idx % args.print_freq == 0:
                    running_avg_loss = total_loss / max(total_samples, 1)
                    running_avg_acc = total_acc / max(total_samples, 1)
                    print(
                        f"Task {task_id+1}, Batch [{batch_idx}/{len(data_loader)}]: "
                        f"Running Avg Loss = {running_avg_loss:.2f}, Running Avg Acc@1 = {running_avg_acc:.2f}"
                    )

        avg_acc = total_acc / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)
        print(f"Task {task_id+1}: Final Avg Loss = {avg_loss:.2f} | Final Avg Acc@1 = {avg_acc:.2f}")
        return float(avg_acc)

    def evaluate_till_now(self, model, data_loader, device, task_id: int, acc_matrix: np.ndarray, args):
        for t in range(task_id + 1):
            acc_matrix[t, task_id] = self.evaluate_task(model, data_loader[t]["val"], device, t, args)

        A_i = [np.mean(acc_matrix[: i + 1, i]) for i in range(task_id + 1)]
        A_last = A_i[-1]
        A_avg = float(np.mean(A_i))

        result_str = f"[Average accuracy till task{task_id+1}] A_last: {A_last:.2f} A_avg: {A_avg:.2f}"
        if task_id > 0:
            forgetting = float(np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id]))
            result_str += f" Forgetting: {forgetting:.4f}"
        else:
            forgetting = 0.0

        if args.wandb:
            import wandb

            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})

        print(result_str)
        if args.verbose or args.wandb:
            sub_matrix = acc_matrix[: task_id + 1, : task_id + 1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            heatmap_path = save_accuracy_heatmap(result, task_id, args)
            if args.wandb:
                import wandb

                wandb.log({"Accuracy Heatmap": wandb.Image(heatmap_path)})
        return {"Acc@1": A_last, "A_avg": A_avg, "Forgetting": forgetting}


def parse_args():
    p = argparse.ArgumentParser("MORE on OOD-VIL (VIL scenario)")

    # data / scenario
    p.add_argument("--dataset", type=str, default="CLEAR", choices=["iDigits", "DomainNet", "CORe50", "CLEAR"])
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--IL_mode", type=str, default="vil", choices=["cil", "dil", "vil", "joint"])
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--shuffle", action="store_true")

    # training
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # backbone / pretrain
    p.add_argument(
        "--backbone",
        type=str,
        default="deit_small_patch16_224",
        choices=["deit_small_patch16_224", "vit_base_patch16_224"],
        help="Paper uses DeiT-S/16 (deit_small_patch16_224).",
    )
    p.add_argument(
        "--more_pretrain_path",
        type=str,
        default="./deit_pretrained/best_checkpoint.pth",
        help="Path to MORE pre-trained DeiT checkpoint (as in README).",
    )
    p.add_argument(
        "--require_more_pretrain",
        action="store_true",
        help="If set, error out when MORE pretrain checkpoint is missing.",
    )

    # MORE / ViT-Adapter / HAT
    p.add_argument("--adapter_latent", type=int, default=64)
    p.add_argument("--freeze_head", action="store_true")
    p.add_argument("--smax", type=float, default=500.0)
    p.add_argument("--lamb0", type=float, default=0.75)
    p.add_argument("--lamb1", type=float, default=0.75)
    p.add_argument("--thres_cosh", type=float, default=50.0)
    p.add_argument("--thres_emb", type=float, default=6.0)
    p.add_argument("--md_c", type=float, default=20.0)
    # Paper uses raw softmax probabilities (no temperature). Keep configurable but default to 1.0.
    p.add_argument("--infer_T", type=float, default=1.0)
    p.add_argument("--md_max_samples_per_class", type=int, default=128)

    # replay buffer (BYTES)
    p.add_argument("--replay_buffer_bytes", type=int, default=0)
    p.add_argument("--replay_batch_size", type=int, default=-1)
    p.add_argument(
        "--replay_store_dtype",
        type=str,
        default="uint8",
        choices=["uint8", "float32"],
        help="How to store replay images inside the byte-budgeted buffer.",
    )

    # back-updating (optional; can be expensive)
    p.add_argument("--back_update", action="store_true")
    p.add_argument("--back_update_epochs", type=int, default=10)
    p.add_argument("--back_update_lr", type=float, default=0.01)
    p.add_argument("--back_update_batch_size", type=int, default=16)
    p.add_argument("--back_update_samples", type=int, default=1024)

    # logging / eval
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--develop", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)

    # wandb (same logic as oodvil code.md)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run", type=str, default=None)

    # optional OOD dataset (kept for parity; not required for core MORE eval)
    p.add_argument("--ood_dataset", type=str, default=None)
    p.add_argument("--ood_eval_samples", type=int, default=0, help="If >0, use this many ID/OOD samples each for AUROC.")
    p.add_argument("--ood_eval_every", type=int, default=1, help="Run OOD eval every N tasks (default: 1).")

    return p.parse_args()


def main():
    args = parse_args()
    args = set_data_config(args)
    device = torch.device(args.device)

    seed_everything(args.seed)

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("./oodvil_runs", f"{args.dataset}_{args.IL_mode}_{ts}")
    os.makedirs(args.output_dir, exist_ok=True)

    # build_continual_dataloader currently overrides args.verbose; keep user intent
    user_verbose = bool(args.verbose)
    data_loader, class_mask, domain_list = build_continual_dataloader(args)
    args.verbose = user_verbose
    if args.ood_dataset:
        data_loader[-1]["ood"] = get_ood_dataset(args.ood_dataset, args)

    # class_mask is required for VIL; normalize to list[list[int]]
    if class_mask is None:
        raise RuntimeError("class_mask is None. Use IL_mode='vil' (OOD-VIL scenario).")
    class_mask = [list(m) for m in list(class_mask)]

    # infer per-task class count (assume constant across tasks)
    num_cls_per_task = len(class_mask[0])
    if any(len(m) != num_cls_per_task for m in class_mask):
        raise RuntimeError("VIL scenario tasks have varying number of classes; this script assumes constant |Y^k|.")
    args.num_cls_per_task = num_cls_per_task

    # build per-task label mapping (global -> local)
    task_class_to_local: List[Dict[int, int]] = []
    for t in range(args.num_tasks):
        task_class_to_local.append({int(c): i for i, c in enumerate(class_mask[t])})

    if args.replay_batch_size is None or int(args.replay_batch_size) < 0:
        args.replay_batch_size = int(args.batch_size)

    print(args)

    # wandb init (same pattern as oodvil code.md)
    args.wandb = False
    if args.wandb_run and args.wandb_project:
        import getpass
        import wandb

        args.wandb = True
        wandb.init(entity="OODVIL", project=args.wandb_project, name=args.wandb_run, config=vars(args))
        wandb.config.update({"username": getpass.getuser()})

    # model (paper backbone: DeiT-S/16)
    if args.backbone == "deit_small_patch16_224":
        build_fn = deit_small_patch16_224
    elif args.backbone == "vit_base_patch16_224":
        build_fn = vit_base_patch16_224
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # If MORE checkpoint exists, load it (paper setting). Otherwise fall back to timm pretrained weights.
    use_timm_pretrained = not os.path.isfile(str(args.more_pretrain_path))
    net = build_fn(
        pretrained=use_timm_pretrained,
        num_classes=args.num_cls_per_task + 1,
        latent=args.adapter_latent,
        args=args,
    )
    if not use_timm_pretrained:
        loaded = load_more_deit_pretrain_if_available(net, args.more_pretrain_path, require=args.require_more_pretrain)
        if loaded:
            print(f"Loaded MORE pretrain from: {args.more_pretrain_path}")
    else:
        if args.require_more_pretrain:
            load_more_deit_pretrain_if_available(net, args.more_pretrain_path, require=True)
        print("MORE pretrain not found; using timm pretrained backbone weights.")
    net.to(device)

    # replay
    replay = ByteReplayBuffer(capacity_bytes=args.replay_buffer_bytes, seed=args.seed, store_dtype=args.replay_store_dtype)

    # inference wrapper
    wrapper = MoreOODVILWrapper(
        net=net,
        class_mask=class_mask,
        num_classes=args.num_classes,
        num_cls_per_task=args.num_cls_per_task,
        smax=args.smax,
        md_c=args.md_c,
        infer_T=args.infer_T,
    ).to(device)

    trainer = TrainerOODVILMore()
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks), dtype=np.float32)

    p_mask = None
    mask_back = None
    ext_auc_hist: List[float] = []

    for task_id in range(args.num_tasks):
        print(f"{f'Training on Task {task_id+1}/{args.num_tasks}':=^60}")

        # add new task-specific embeddings/head
        net.append_embedddings()
        set_requires_grad_for_more(net, freeze_head=args.freeze_head)
        prepare_hat(net, mask_back)

        # reset optimizer (SGD as in MORE settings)
        optimizer = SGD_hat(
            [p for p in net.adapter_parameters() if p.requires_grad],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        train_start = time.time()
        for epoch in range(args.epochs):
            epoch_start = time.time()
            epoch_avg_loss, epoch_avg_acc = trainer.train_one_epoch(
                net=net,
                task_id=task_id,
                data_loader=data_loader[task_id]["train"],
                optimizer=optimizer,
                device=device,
                class_to_local=task_class_to_local[task_id],
                replay=replay,
                p_mask=p_mask,
                args=args,
            )
            epoch_duration = time.time() - epoch_start
            print(
                f"Epoch [{epoch+1}/{args.epochs}] Completed in {str(datetime.timedelta(seconds=int(epoch_duration)))}: "
                f"Avg Loss = {epoch_avg_loss:.4f}, Avg Acc@1 = {epoch_avg_acc:.2f}"
            )
            if args.wandb:
                import wandb

                wandb.log(
                    {
                        "Train Loss": epoch_avg_loss,
                        "Train Acc@1": epoch_avg_acc,
                        "Epoch": epoch,
                        "TASK": task_id,
                    }
                )
        train_duration = time.time() - train_start
        print(f"Task {task_id+1} training completed in {str(datetime.timedelta(seconds=int(train_duration)))}")

        # compute MD stats for this task (for task-free inference)
        means, cov_inv = compute_task_stats(
            net=net,
            task_id=task_id,
            train_loader=data_loader[task_id]["train"],
            class_to_local=task_class_to_local[task_id],
            num_cls_per_task=args.num_cls_per_task,
            smax=args.smax,
            device=device,
            max_samples_per_class=args.md_max_samples_per_class,
        )
        wrapper.set_task_stats(task_id, means=means, cov_inv=cov_inv)
        wrapper.set_learned_tasks(task_id + 1)

        # end-task: update HAT masks
        p_mask = cum_mask(net, task_id, p_mask, smax=args.smax, device=device)
        mask_back = freeze_mask(net, p_mask)

        # end-task: update replay memory (byte-budgeted)
        replay.update_from_loader(data_loader[task_id]["train"], task_id=task_id, new_classes=class_mask[task_id])
        if args.wandb:
            import wandb

            wandb.log(
                {
                    "Replay bytes used": replay.bytes_used(),
                    "Replay capacity bytes": args.replay_buffer_bytes,
                    "Replay num samples": len(replay),
                    "TASK": task_id,
                }
            )

        # back-updating phase (MORE)
        if args.back_update and task_id > 0:
            print(f"{f'Back-updating previous heads (after Task {task_id+1})':=^60}")
            bu_start = time.time()
            back_update_previous_heads(
                net=net,
                current_task_id=task_id,
                replay=replay,
                current_train_loader=data_loader[task_id]["train"],
                class_mask=class_mask,
                task_class_to_local=task_class_to_local,
                args=args,
                device=device,
            )
            bu_dur = time.time() - bu_start
            print(f"Back-updating completed in {str(datetime.timedelta(seconds=int(bu_dur)))}")

        # evaluate (same A_last/A_avg/Forgetting logic as oodvil code.md)
        print(f'{f"Testing on Task {task_id+1}/{args.num_tasks}":=^60}')
        eval_start = time.time()
        trainer.evaluate_till_now(wrapper, data_loader, device, task_id, acc_matrix, args)
        eval_duration = time.time() - eval_start
        print(f"Task {task_id+1} evaluation completed in {str(datetime.timedelta(seconds=int(eval_duration)))}")

        # OOD detection evaluation
        do_ood_eval = int(args.ood_eval_every) > 0 and ((task_id + 1) % int(args.ood_eval_every) == 0)
        if do_ood_eval:
            id_dataset = torch.utils.data.ConcatDataset([data_loader[t]["val"].dataset for t in range(task_id + 1)])

            # (1) External OOD dataset (user-provided, e.g., EMNIST)
            if args.ood_dataset:
                ood_ds = data_loader[-1].get("ood", None)
                if ood_ds is not None:
                    print(f"{f'OOD Detection (external: {args.ood_dataset})':=^60}")
                    evaluate_ood_detection(
                        model=wrapper,
                        id_dataset=id_dataset,
                        ood_dataset=ood_ds,
                        device=device,
                        args=args,
                        task_id=task_id,
                        tag=f"EXT_{args.ood_dataset}",
                        auc_history=ext_auc_hist,
                    )


if __name__ == "__main__":
    main()

