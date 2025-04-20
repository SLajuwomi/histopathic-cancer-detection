# Histopathologic Cancer Detection - PyTorch TPU Solution

## Overview

This project implements a simple solution for the Kaggle "Histopathologic Cancer Detection" competition using PyTorch and Google Colab TPUs. It utilizes transfer learning with a pretrained EfficientNet-B0 model to classify 96x96 pixel image patches from lymph node scans as either containing metastatic cancer tissue (label 1) or not (label 0).

## Assignment Context

This project fulfills the requirements for **Option 2** of the Neural Network assignment:
*   Uses a major framework (PyTorch).
*   Targets the specified Kaggle Histopathologic Cancer Detection dataset.
*   Employs transfer learning from a pretrained model (EfficientNet-B0) with added trainable layers (the final classifier).
*   Includes Exploratory Data Analysis (EDA), model creation, training, evaluation (including AUC metric), discussion of learning rate (via comments/basic setup), and generates a submission file.

## Technology Used

*   Python 3
*   PyTorch
*   `torch_xla` for TPU support
*   Torchvision (for models and transforms)
*   EfficientNet-B0 (pretrained on ImageNet)
*   Pandas (for data manipulation)
*   Matplotlib (for plotting)
*   Scikit-learn (for train/test split and AUC calculation)
*   Kaggle API (for dataset download)
*   Google Colab (with TPU runtime)

## Getting Started

### Prerequisites

1.  A Google Account with access to Google Colab (Colab Pro recommended for reliable TPU access).
2.  Your Kaggle API key (`kaggle.json` file). You can download this from your Kaggle account settings page.

### How to Run

1.  **Upload Notebook:** Upload the `pytorch_histopath_tpu.ipynb` notebook to your Google Colab environment.
2.  **Set Runtime:** In Colab, go to `Runtime` -> `Change runtime type` and select `TPU` as the Hardware accelerator.
3.  **Upload Kaggle Key:** Run the initial setup cells in the notebook. You will be prompted to upload your `kaggle.json` file.
4.  **Run All:** Execute the remaining cells sequentially (you can use `Runtime` -> `Run all`). The notebook will:
    *   Install necessary libraries.
    *   Download and prepare the dataset.
    *   Perform basic EDA.
    *   Define and load the model.
    *   Train the model on the TPU for a few epochs.
    *   Evaluate the model and plot results.
    *   Generate a `submission.csv` file.

## Notebook Structure

The Jupyter notebook (`pytorch_histopath_tpu.ipynb`) is organized as follows:

1.  **Setup:** Installs libraries, imports modules, handles Kaggle API setup, downloads data, and initializes the TPU device.
2.  **Constants & Paths:** Defines global parameters like image size, batch size, and file paths.
3.  **Exploratory Data Analysis (EDA):** Loads labels, checks distribution, and visualizes sample images.
4.  **Data Preparation:** Splits data, defines image transformations, creates a custom PyTorch `Dataset`, and sets up `DataLoaders`.
5.  **Model Definition:** Loads a pretrained EfficientNet-B0, freezes base layers, and replaces the final classifier layer.
6.  **Training & Validation:** Defines optimizer, loss function, and the main training/validation loop compatible with `torch_xla`.
7.  **Evaluation & Results:** Plots learning curves (loss, accuracy, AUC) and prints final metrics.
8.  **Submission File Generation:** Runs inference on the test set and creates `submission.csv`.
9.  **Conclusion:** Summarizes the process and suggests potential improvements.

## Output

*   Training and validation metrics printed during execution.
*   Plots showing loss and validation metrics over epochs.
*   A `submission.csv` file in the Colab environment, ready for download or submission.
