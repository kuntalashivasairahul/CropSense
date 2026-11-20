# Notebook index

All **nine** existing notebooks are included. Read the [team Kaggle guide](../docs/team/KAGGLE_GUIDE.md) for data access, version history, evidence, and safe reruns.

| Module | Current audit source | Original export with saved outputs | Older Kaggle source snapshot |
|---|---|---|---|
| Leaf disease | [Module 1 audit](cropsense-module-1-audit/cropsense-module-1-audit.ipynb) | [Original 1](originals/module-1.ipynb) | [Snapshot 1](kaggle-snapshots/module-1/module-1.ipynb) |
| Canopy stress | [Module 2 audit](cropsense-module-2-audit/cropsense-module-2-audit.ipynb) | [Original 2](originals/module-2.ipynb) | [Snapshot 2](kaggle-snapshots/module-2/module-2.ipynb) |
| Pest risk | [Module 3 audit](cropsense-module-3-audit/cropsense-module-3-audit.ipynb) | [Original 3](originals/module-3.ipynb) | [Snapshot 3](kaggle-snapshots/module-3/module-3.ipynb) |

Each audit folder and each snapshot folder has its original `kernel-metadata.json`. Audit sources contain **17/16/16 cells** and no embedded outputs. Snapshots also have no embedded outputs. Original exports preserve their historical outputs; their original hashes are recorded in [notebook_sources.json](../docs/evidence/notebook_sources.json).

The audit notebooks are the main reading path. Originals and snapshots preserve development history and may use different models, features, targets, splits, or metrics. They must not silently replace current runtime contracts. A checked-in source copy can include changes made after a hosted saved run; its code does not prove that a recorded result was rerun.
