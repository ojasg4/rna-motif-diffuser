# RNA-Motif-Diffuser (RMD)

RNA-Motif-Diffuser (RMD) is a hybrid CNN–Transformer–diffusion architecture for
learning RNA sequence–structure relationships from large-scale datasets.

RMD combines:

- A 1D CNN + Transformer encoder for sequence and motif-aware embeddings
- An Evoformer-style pairwise interaction module for long-range contacts
- An SE(3)-equivariant diffusion model over 3D coordinates
- A kinematic loop refiner for geometric cleanup

The codebase is organized for use on a single GPU (e.g., Kaggle) and multi-GPU
HPC clusters.

---

## Features

- Motif-aware sequence encoder (CNN + Transformer)
- Evoformer-inspired pair representation
- SE(3)-equivariant diffusion over 3D coordinates
- Optional coordinate normalization and recentering
- Single-GPU and multi-GPU (DDP) support
- Modular design for ablations and extension

---

## Installation

```bash
git clone https://github.com/<your-username>/rna-motif-diffuser.git
cd rna-motif-diffuser

# Recommended: create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
