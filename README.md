# IPA_Project_Implementation

A PyTorch-based implementation for low-light image enhancement using a Multi-Branch Progressive Network (MBPN) within a Generative Adversarial Network (GAN) framework. [1](#0-0) 

## Overview

This project addresses the challenge of enhancing images captured in extremely low-light conditions through iterative refinement using a GAN architecture where the Generator is an MBPN model that processes input images through multiple progressive stages. [2](#0-1) 

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
The training script manages the training loop across epochs, updates learning rates, and triggers periodic validation through the `envclass`. [3](#0-2) 

### Testing
```bash
python test.py
```
The test script loads a trained "bestmode" checkpoint and generates enhanced outputs for the test dataset. [4](#0-3) 

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
The main model class that encapsulates the GAN training lifecycle, including the Generator and NLayerDiscriminator. [5](#0-4) 

### Data Pipeline
Expects a directory structure with `trainlow`, `trainhigh`, `testlow`, and `testhigh` folders for paired low-light and ground-truth images. [6](#0-5) 

### Evaluation System
The `envclass` handles validation logic, calculates metrics, and determines when to save the best model based on validation performance. [7](#0-6) 

## Notes

The current README.md contains only the project title. [1](#0-0)  This comprehensive README is generated based on the project's wiki overview and codebase structure. For detailed architectural information, refer to the Project Overview wiki page which contains extensive documentation about the MBPN architecture, training procedures, and component relationships.

Wiki pages you might want to explore:
- [Project Overview (molly499/IPA_Project_Implementation)](/wiki/molly499/IPA_Project_Implementation#1)

### Citations

**File:** README.md (L1-1)
```markdown
# IPA_Project_Implementation
```

**File:** train.py (L7-7)
```python
from models.ours_model import MBPNModel
```

**File:** train.py (L13-13)
```python
    dataset = create_dataset(opt)  # Import dataset to generate dataset path collection
```

**File:** train.py (L15-16)
```python
    model = MBPNModel(opt)      # Create models
    model.setup(opt)               # Learning rate setting startup model
```

**File:** train.py (L22-56)
```python
    for epoch in range(opt.epoch_count, opt.maxepoch + 1):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0  #The number of iterations in the epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:  #Show the frequency of intermediate results of training
                visualizer.display_current_results(model.get_current_visuals(), total_iters)

            if total_iters % opt.print_freq == 0:#Print current loss
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, model.get_current_losses(), t_comp)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0: #saves the model and test
            print('saving the model at the end of epoch %d and latest, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

            savebest = myenv.env(epoch) #test
            if savebest == "bestmode":
                model.save_networks('bestmode')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.maxepoch, time.time() - epoch_start_time))
        model.update_learning_rate()#Update learning rate
```

**File:** test.py (L7-13)
```python
if __name__ == '__main__':

    opt = BaseOptions().parse()

    myenv=envclass(opt)

    myenv.envsave("bestmode")
```
