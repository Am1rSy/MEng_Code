# Correcting Off-Resonance Artefacts in Short-Lived Tissue

This repository contains the codebase developed for a Master's thesis project focused on correcting off-resonance artefacts in ultrashort echo time (UTE) MRI. The project leverages synthetic data generation and deep learning techniques to address challenges associated with non-Cartesian k-space sampling, specifically radial and spiral trajectories.

## Repository Structure

### 1. Synthetic Data Generation

- **`data_generator.ipynb`**  
  Jupyter notebook for generating synthetic MRI images using off-resonance artefact simulations.  
  This notebook calls helper functions defined in `SyntheticDataGenerator_sharp`.

- **`SyntheticDataGenerator_sharp.py`**  
  Python script containing the main routines to simulate frequency maps, off-resonance phase effects, and k-space data from synthetic anatomical models.

---

### 2. Model Training (Spiral Trajectory)

- **`modl_offres_spiral.ipynb`**  
  Trains the deep learning model using a spiral k-space sampling trajectory.  
  This version **uses** the [`torchkbnufft`](https://github.com/mmuckley/torchkbnufft)  library to perform non-uniform fast Fourier transform (NUFFT) operations, enabling more realistic modelling of spiral acquisition and off-resonance artefacts.
  
---
### 3. Model Training (Pseudo-Radial Trajectory)

- **`modl_offres_radial.ipynb`**  
  Trains the deep learning model using a pseudo-radial k-space sampling strategy.  
  This version does **not** use the `torchkbnufft` library and instead employs a simplified regridding method for radial sampling.

