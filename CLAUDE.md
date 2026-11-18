<!-- Parent: ../CLAUDE.md -->

> **Instruction chain:** Read the parent [`../CLAUDE.md`](../CLAUDE.md) before this file; its broader instructions also apply to this subtree.

<!-- Generated: 2026-04-23 | Updated: 2026-05-19 -->

# MORE

## Purpose
**MORE** (Kim & Liu, CoLLAs'22) — multi-head CIL with OOD replay. Conceptually similar to the proposed ETE (multi-head + OOD training) but adapted here to the OOD-VIL protocol. **Dropped from the ECCV'26 main table** (seeds incomplete). Status: future-work / discussion only.

## Key Files
| File | Description |
|------|-------------|
| `main_ood_vil.py` | OOD-VIL entrypoint (adapted from the upstream `main.py`). |
| `run.py` | Upstream run wrapper. |
| `training.py`, `train_clf.py` | Training / classifier fit loops. |
| `testing.py` | Evaluation loop. |
| `common.py` | Shared utilities. |
| `custom_imagefolder.py` | Dataset helper. |
| `README.md` | Upstream README. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `apprs/` | MORE algorithm modules (multi-head logic, OOD replay). |
| `clip/` | CLIP-backbone variant. |
| `networks/` | ViT-HAT and other network defs. **⚠️ `networks/my_vit_hat.py:654` has the timm `default_cfg` bug — patch before running.** |
| `continual_datasets/` | iDigits / DomainNet loaders + OOD shims. |
| `datasets/`, `data/` | Dataset caches / download scripts. |
| `utils/` | Logger / seed / wandb glue. |
| `integration/` | OOD-VIL-specific integration glue (added by us). |

## For AI Agents

### Working In This Directory
- **⚠️ timm bug**: `TypeError: __init__() got an unexpected keyword argument 'default_cfg'` at `networks/my_vit_hat.py:654`. Fix options: (a) pin `timm` to a version that still accepts `default_cfg`, (b) patch the call to use `pretrained_cfg`. **Must fix before any MORE run.**
- **Not in ECCV main table** — only run if the user explicitly asks to patch / complete seeds for future work.
- **Run template** (iDigits seed 5 missing, per `experiments/TODO.md`):
  ```bash
  python main_ood_vil.py --dataset iDigits --IL_mode vil --num_tasks 20 \
      --ood_dataset EMNIST --data_path /local_datasets \
      --epochs 10 --batch_size 32 --replay_buffer_bytes 30720000 \
      --back_update --backbone vit_base_patch16_224 \
      --wandb_project OODIL_iDigits_Maintable --wandb_run MORE --seed 5
  ```

### Testing Requirements
- Smoke-test with a tiny `--num_tasks 3` run after timm patch to confirm fix works.
- Reproduction project: `OODVIL/reproduce_check`.

### Common Patterns
- Replay buffer: `--replay_buffer_bytes 30720000` (matches main-table budget).
- wandb entity: `OODVIL`.

## Dependencies

### Internal
- `experiments/TODO.md` (outside) — seed 5 pending.
- `docs/baselines.md`, `docs/paper_status.md` — dropped-from-main-table reasoning.

### External
- PyTorch, timm (version-sensitive), wandb.
- GitHub remote: `guhyun9454/MORE`, branch `integrate_OODVIL`.

<!-- MANUAL: -->
