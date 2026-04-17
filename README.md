# IPA_Project_Implementation

A PyTorch-based implementation for low-light image enhancement using a Multi-Branch Progressive Network (MBPN) within a Generative Adversarial Network (GAN) framework. 

## Overview

This project addresses the challenge of enhancing images captured in extremely low-light conditions through iterative refinement using a GAN architecture where the Generator is an MBPN model that processes input images through multiple progressive stages. 

## Key Features

- **Multi-Branch Progressive Network (MBPN)**: Core generator architecture with encoder-decoder structure and LSTM cells for state maintenance across refinement steps
- **GAN Framework**: Includes both generator and PatchGAN discriminator for adversarial training
- **Comprehensive Evaluation**: Supports PSNR, SSIM, and LPIPS metrics for quality assessment
- **Progressive Training**: Iterative refinement through multiple stages with validation-based model saving

## Quick Start

### Training
```bash
python train.py
```
The training script manages the training loop across epochs, updates learning rates, and triggers periodic validation through the `envclass`.

### Testing
```bash
python test.py
```
The test script loads a trained "bestmode" checkpoint and generates enhanced outputs for the test dataset.

## Project Structure

```
├── train.py              # Training entry point
├── test.py               # Testing entry point  
├── models/
│   └── ours_model.py     # MBPNModel implementation
├── data/
│   └── ours_dataset.py   # Dataset handling
├── options/
│   └── base_options.py   # Configuration parsing
├── util/
│   └── visualizer.py     # Training visualization
└── envclass.py           # Evaluation and validation
```

## Core Components

### MBPNModel
The main model class that encapsulates the GAN training lifecycle, including the Generator and NLayerDiscriminator.

### Data Pipeline
Expects a directory structure with `trainlow`, `trainhigh`, `testlow`, and `testhigh` folders for paired low-light and ground-truth images.

### Evaluation System
The `envclass` handles validation logic, calculates metrics, and determines when to save the best model based on validation performance.
