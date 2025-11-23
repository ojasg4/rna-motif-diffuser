# RNA-Motif-Diffuser (RMD)

This is a hybrid CNN–Transformer–diffusion architecture for learning RNA sequence–structure relationships from large-scale datasets. RMD combines:

- A 1D CNN + Transformer encoder for sequence and motif-aware embeddings
- An Evoformer-style pairwise interaction module for long-range contacts
- An SE(3)-equivariant diffusion model over 3D coordinates
- A kinematic loop refiner for geometric cleanup

The data preprocessing script was executed using a kaggle dataset: https://www.kaggle.com/datasets/jaejohn/rna-all-data.

Originally provided by Professor Rhiju Das and filtered by kaggle user John G Rao (@jaejohn) as part of the Stanford 3D RNA folding competition.

Training is organized for debugging on a single GPU (Kaggle, or locally) and training on multi-GPU HPC clusters.

---

## Features

- Motif-aware sequence encoder (CNN + Transformer)
- Evoformer-inspired pair representation
- SE(3)-equivariant diffusion over 3D coordinates
- Optional coordinate normalization and recentering
- Single-GPU and multi-GPU (DDP) support
- Modular design for ablations and extension

