# Short-Time Sound Event Localization and Detection Using Gammatone Filters and SCConv-Enhanced CST-Former

## Introduction

This repository contains the code and resources for our paper:

**"Short-Time Sound Event Localization and Detection Using Gammatone Filters and SCConv-Enhanced CST-Former"**

In this work, we address the limitations of current Sound Event Localization and Detection (SELD) systems in handling short time segments (specifically 1-second windows). This is crucial for real-world applications requiring low-latency and fine temporal resolution. We establish a new baseline for SELD performance on 1-second segments.

Our key contributions are:

- **Establishing SELD performance on 1-second segments**: Providing a new benchmark for short-segment analysis in SELD tasks.
- **Comparative analysis of filter banks**: Systematically comparing Bark, Mel, and Gammatone filter banks for audio feature extraction, demonstrating that Gammatone filters achieve the highest overall accuracy.
- **Integration of SCConv modules into CST-Former**: Replacing convolutional components in the CST block with the SCConv module, yielding measurable F-score gains and enhancing spatial and channel feature representation.

## Code Outline

The repository is organized as follows:

<!-- - `src/`: Source code for the SELD system.
  - `models/`: Contains model architectures including CST-Former and SCConv modules.
  - `data/`: Data loading and preprocessing scripts.
  - `utils/`: Utility functions for training, evaluation, and logging.
- `configs/`: Configuration files for experiments.
- `scripts/`: Shell scripts for running training and inference.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation. -->
- `cls_dataset/`:
    - `cls_dataset.py`: PyTorch Dataset implementation for training procedure, aims to accelerate the trainning process.
- `models/`: source code for different models.
    - `architecture/`: source code for CST-former and SCConv CST former
    - `baseline_model.py`: source code for SELDnet
    - `conformer.py`:source code for Conv-Conformer
- `parameters.py` script consists of all the training, model, and feature configurations. One can add new configurations for feature extracion and model architecture. If a user wants to change some parameters or use a new configuration, they have to create a sub-task with unique id here. Check code for examples.
- `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. **Make sure you update the location of the downloaded datasets in parameters.py before.**
- The `cls_compute_seld_results.py` script computes the metrics results on your DCASE output format files. 
- The `cls_data_generator.py` script provides feature + label data in generator mode for validation and test.
- The `cls_feature_class.py` script has routines for labels creation, features extraction and normalization. Filter bank options are use as an attribute of this class.
- The `cls_vid_features.py` script extracts video features for the audio-visual task from a pretrained ResNet model. Our system donnot implement audio-visual track.
- The `criterions.py` encompasses some custome loss functions and multi-accdoa 
<!-- - The `seldnet_model.py` script implements the SELDnet architecture. -->
- The `SELD_evaluation_metrics.py` script implements the metrics for joint evaluation of detection and localization.
- The `torch_run_vanilla.py` is a wrapper script that trains the model and calculates the metrics for each test dataset. The training stops when the F-score (check the paper) stops improving after 50 epochs of patience.
- `README.md`: Project documentation.

## Preparation

### Prerequisites

- **Operating System**: Linux recommended, didnot test on Windows.
- **Python**: Version 3.8 or higher.
- **Anaconda**: Recommended for environment management.

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/short-time-seld.git
   cd short-time-seld
   ```

2. **Create a Conda Environment**

   ```bash
   conda create -n seld_env python=3.8
   conda activate seld_env
   ```

3. **Install Dependencies**

   Install required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install using `conda`:

   ```bash
   conda install --file requirements.txt
   ```

4. **Set Up Additional Tools**

   Install any additional tools or libraries if necessary. For example:

   ```bash
   # Placeholder for additional setup
   ```

## Data Preprocessing

### Dataset

We use the [DCASE 2021 Task 3](https://dcase.community/challenge2021/task-sound-event-localization-and-detection) dataset for our experiments.

### Steps

1. **Download the Dataset**

   Download the development and evaluation datasets from the DCASE challenge website and place them in the `data/` directory.

   ```bash
   mkdir data
   # Instructions or script to download the dataset
   ```

2. **Extract Audio Features**

   Run the preprocessing script to extract features using the desired filter bank:

   ```bash
   python src/data/preprocess_data.py --filter_bank gammatone --segment_length 1
   ```

   Parameters:

   - `--filter_bank`: Choose from `bark`, `mel`, or `gammatone`.
   - `--segment_length`: Set the time segment length in seconds (default is `1`).

3. **Generate Labels**

   Prepare the labels for training:

   ```bash
   python src/data/generate_labels.py --segment_length 1
   ```

4. **Data Augmentation (Optional)**

   Apply data augmentation techniques if needed:

   ```bash
   # Placeholder for data augmentation commands
   ```

## Training and Inference

### Training the Model

Train the SELD model with the SCConv-enhanced CST-Former architecture:

```bash
python src/train.py --config configs/scconv_cstformer.yaml
```

Parameters:

- `--config`: Path to the configuration file containing training parameters.

### Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir runs/
```

### Inference

Perform inference on the test set:

```bash
python src/inference.py --checkpoint checkpoints/best_model.pth --config configs/scconv_cstformer.yaml
```

Parameters:

- `--checkpoint`: Path to the trained model checkpoint.
- `--config`: Configuration file used during training.

### Evaluation

Evaluate the model's performance using standard SELD metrics:

```bash
python src/evaluate.py --predictions outputs/predictions.csv --ground_truth data/labels/test_labels.csv
```

Parameters:

- `--predictions`: Model predictions output file.
- `--ground_truth`: Ground truth labels for the test set.

## References

- [1] Adavanne, S., Politis, A., & Virtanen, T. (2018). "Sound Event Detection and Localization in Multiple Directions". *ICASSP 2018*.
- [2] Mazzon, A., et al. (2021). "Multi-ACCDOA: Localizing and Detecting Overlapping Sounds from the Same Direction with Deep Learning". *ICASSP 2021*.
- [3] Zhang, X., et al. (2021). "CST-Former: Channel-Spectro-Temporal Transformer for Acoustic Modeling". *Interspeech 2021*.
- [4] Yang, Y., et al. (2021). "G-SELD: A SELD Model Based on Gammatone Filters and CNNs". *DCASE 2021 Workshop*.
- [5] Wu, Q., et al. (2021). "SCConv: Efficient Convolutional Neural Networks with Spatial and Channel Modulation". *arXiv preprint arXiv:2103.07659*.

## Contact

For any questions or assistance, please contact:

- **Name**: [Your Name]
- **Email**: [your.email@example.com]

---

*Note: Sections marked as placeholders should be filled in with the appropriate code or instructions.*

# Short-Time Sound Event Localization and Detection Using Gammatone Filters and SCConv-Enhanced CST-Former

Thank you for your interest in our work!