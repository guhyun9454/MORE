<!-- Parent: ../CLAUDE.md -->

> **Instruction chain:** Read the parent [`../CLAUDE.md`](../CLAUDE.md) before this file; its broader instructions also apply to this subtree.

<!-- Generated: 2026-05-03 | Updated: 2026-05-19 -->

# MORE/integration

## Purpose
Design documents for the MORE × OOD-VIL integration. Records what changes were made to adapt the upstream MORE codebase to the OOD-VIL evaluation protocol.

## Key Files
| File | Description |
|------|-------------|
| `MORE paper.md` | Summary of the upstream MORE paper (CoLLAs'22) — method, claims, and relevance to OOD-VIL. |
| `oodvil code.md` | Code-level notes on OOD-VIL integration: which files were changed, what the `integrate_OODVIL` branch adds, and known issues. |

## For AI Agents

### Working In This Directory
- These are **read-only reference documents** — do not edit.
- **MORE is dropped from the ECCV main table** (seeds incomplete, timm `default_cfg` bug). It is future-work status.
- Before running any MORE job, read `oodvil code.md` for the `TypeError: __init__() got an unexpected keyword argument 'default_cfg'` fix needed in `networks/my_vit_hat.py`.

## Dependencies

### Internal
- `networks/my_vit_hat.py` — where the timm bug manifests.
- `baseline_repo/Unified_OOD_CIL/CLAUDE.md` — MORE status and known issues.

<!-- MANUAL: -->
