# Video Classification with UCF50

This project implements a video classification pipeline using the UCF50 dataset. It leverages a Long-term Recurrent Convolutional Network (LRCN) model that extracts spatial features from individual video frames via a ConvNeXt-Tiny backbone and learns temporal dynamics through an LSTM. The project includes scripts for preprocessing, training, and testing the model.

---

## Table of Contents

- [Errors Found and Fixed](#errors-found-and-fixed)
- [Model Improvements](#model-improvements)
- [Results](#results)
- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Preprocessing and Frame Extraction](#preprocessing-and-frame-extraction)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [Unit Testing and Code Quality](#unit-testing-and-code-quality)
- [Project Structure](#project-structure)
- [Customization and Hyperparameters](#customization-and-hyperparameters)

---

## Errors Found and Fixed

### 1. Incorrect LSTM Batch Layout

The original LSTM used its default sequence-first layout, but the forward pass supplied features shaped as `(batch, 1, features)`. This prevented the model from treating each video as a separate sequence of frames. To fix this, LSTM was updated to use `batch_first=True`.

### 2. Data Leakage Between Related Videos

The original split used stratified sampling on individual videos. UCF50 groups contain clips that may share the same person, background, or camera viewpoint, so closely related videos could appear in more than 1 set (i.e., training, validation, testing). This could make validation and test results overly optimistic. To fix this, the split now reads the group number from each video name and uses `GroupShuffleSplit`. Each group stays in only one of training, validation, or testing.

### 3. Frame Order, Padding, and Augmentation

- **Frame order:** Frame filenames are sorted by their numeric index. This prevents `frame10.jpg` from appearing before `frame2.jpg` and preserves chronological order.
- **Requested frame count:** Frame extraction previously selected `n_frames + 1` positions. It now requests `n_frames` positions.
- **Padding:** Real clip lengths are recorded before padding. Padded recurrent outputs are excluded from temporal averaging, and the CNN processes only genuine frames, preventing padded images from affecting ResNet BatchNorm statistics.
- **Consistent augmentation:** Random transforms are applied to the entire clip instead of separately to each frame. This prevents frames of the same clip being inconsistent with each other due to the applied transforms being uniquely applied to each frame.

---

## Model Improvements

### 1. Pretrained ConvNeXt-Tiny Backbone

Backbone experiments showed that ConvNeXt-Tiny achieved higher validation and test accuracy than ResNet-34 and ResNet-50, so it was selected for the final model. Its pretrained CNN extracts features from each video frame, which the recurrent network then uses to model motion over time.

### 2. Bidirectional LSTM

The training script enables `--bidirectional`. The LSTM processes each video's frame features in chronological and reverse order, allowing its representations to incorporate context from both earlier and later frames.

### 3. Layer Normalization Before Temporal Pooling

The training script enables `--rnn_layer_norm`. Layer normalization normalizes each timestep's recurrent features independently, before the model averages features across the video's valid timesteps. This keeps the feature scale more consistent during pooling.

The resulting flow is:

```text
Video frames -> CNN features -> BiLSTM -> LayerNorm -> Mean over valid timesteps -> Dropout -> Class scores
```

### Efforts to Reduce Overfitting

Training uses random temporal segment sampling and spatial/color augmentation shared across the frames in each clip. Random temporal sampling and spatial/color augmentation expose the model to different versions of each training video, reducing its tendency to memorize specific frames or visual details. Validation and test sampling remain deterministic.

---

## Results

### Best Completed Run

The best completed run so far is `20261028_205151_248095`.

| Metric | Result |
|---|---:|
| Test loss | 0.346291 |
| Test accuracy | 92.2090% |

This run was not logged to W&B. Its results are documented using the saved training history/logs, dataset splits, checkpoint, and test-metrics JSON file.

#### Training Configuration for the Best Run

To use the best-run settings reconstructed from the historical training script, replace the contents of `train.sh` with:

```bash
python run.py \
    --frame_dir UCF50_Captured_Frames \
    --train_size 0.60 \
    --test_size 0.20 \
    --cnn_backbone convnext_tiny \
    --pretrained true \
    --rnn_hidden_size 100 \
    --rnn_n_layers 1 \
    --bidirectional \
    --rnn_layer_norm \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 6 \
    --mode train \
    --n_epochs 100 \
    --num_workers 4 \
    --dropout 0.3 \
    --learning_rate 3e-5 \
    --cnn_learning_rate 3e-5 \
    --weight_decay 0.0 \
    --label_smoothing 0.0 \
    --random_temporal_sampling
```

From the project directory, run:

```bash
bash train.sh
```

The epoch limit is 100; completed epochs are recorded in `metrics.csv`.

Matching the settings does not guarantee identical metrics because initialization, shuffled batches, frame sampling, and augmentation include randomness.

#### Evaluate the Best Checkpoint

To display the best checkpoint's test loss and accuracy, replace the contents of `test.sh` with the following:

```bash
python run.py \
    --ckpt ./runs/20261028_205151_248095/models/best_model_wts.pt \
    --cnn_backbone convnext_tiny \
    --rnn_hidden_size 100 \
    --rnn_n_layers 1 \
    --bidirectional \
    --rnn_layer_norm \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 16 \
    --mode eval
```

Keep `splits.npy` in `runs/20261028_205151_248095/`, alongside the `models` folder. From the project directory, run:

```bash
bash test.sh
```

#### Saved Run Artifacts

The following paths refer to the best-run files under `runs/20261028_205151_248095/`:

- [Best model weights](runs/20261028_205151_248095/models/best_model_wts.pt): the saved CNN and recurrent-model weights.
- [Training metrics](runs/20261028_205151_248095/metrics.csv): training and validation loss and accuracy by epoch.
- [Dataset splits](runs/20261028_205151_248095/splits.npy): the video paths and labels assigned to each split.
- [Test metrics](runs/20261028_205151_248095/test_metrics_best_model_wts.json): the measured test loss, test accuracy, and evaluation details.

### W&B Completed Run



---

## Dataset Preparation

### Step 0: Download and Unzip Dataset

1. **Download Dataset:**  
   Download the UCF50 dataset from [UCF](https://www.crcv.ucf.edu/data/UCF50.php). This dataset contains videos of 50 different human action classes.

2. **Unzip and Organize:**  
   Unzip the downloaded dataset. The expected folder structure should be as follows:
   
        - UCF50
            - Action_Class1
            - Action_Class2
            ... ... ... ...
            - Action_Class50

Each subdirectory represents a different action class.

---

## Environment Setup

1. **Python Version:**  
This project requires Python 3.7 or higher.

2. **Dependencies:**  
Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### Key Libraries

- **PyTorch**
- **torchvision**
- **OpenCV**
- **scikit-learn**
- **tqdm**
- **numpy**
- **Pillow**
- **wandb**

## Hardware Requirements

A CUDA-enabled GPU is recommended for training. The code automatically detects GPU availability.

---

## Preprocessing and Frame Extraction

Before training, frames from each UCF50 video need to be captured for the model to learn from. Run preprocess.py to capture frames from the UCF50 videos. The preprocess.py source code uses OpenCV to get frames at evenly spaced positions throughout each video, depending on the number of frames you want to capture, and saves them as JPEG images.

From the project directory, run:
```bash
python preprocess.py --dataset-dir UCF50 --output-dir UCF50_Captured_Frames --n-frames 64
```

This command requests 64 frames per UCF50 video. Some videos produce fewer frames because they are short or contain unreadable frames. The output follows this structure:
```text
UCF50_Captured_Frames/
    BaseballPitch/
        v_BaseballPitch_g01_c01/
            frame0.jpg
            frame1.jpg
            ...
```

### Captured Frames and Model Input

The number of captured frames is separate from the number of frames used by the model. With 64 captured frames and --fr_per_vid 16, keep the following in mind:
* Training: When --random_temporal_sampling is enabled, the dataset divides the 64 frames into 16 consecutive groups and randomly selects one frame from each group, for a total of 16 frames for training for a video. A new selection is made per video loaded.
* Validation/Testing: The dataset will select 16 fixed, uniformly spaced frames for repeatable evaluation.

### Missing Frames After Preprocessing

After running the preprocess.py script, some videos may produce fewer than requested. Videos with no captured frames are skipped. If a video contains fewer than 16 frames, the DataLoader pads it as needed for batching; the model uses the recorded valid number of captured frames to exclude padding from processing and pooling.

Use the --debug flag to the preprocessing command to save progress details in a debug log.

---

## Training the Model

### Step 1: Run Training

#### Configure Training Parameters

The training is managed via a bash script (e.g., `train.sh`) that runs run.py in training mode.

**Important**: Set --frame_dir in train.sh to the directory containing your extracted frames. If you used the preprocessing command above, this directory is UCF50_Captured_Frames.

You can also adjust --fr_per_vid, --batch_size, --num_workers, and the learning rates inside train.sh. The --fr_per_vid setting controls how many frames the model uses per video, rather than how many frames preprocessing saves.

#### Run the Training Script

Execute the training script from your terminal:

```bash
bash train.sh
```

### During Training, the Script Will:

- **Load the captured video frames** from the directory specified by --frame_dir
- **Split the dataset** into training, validation, and test sets using group-disjoint sampling. UCF50 videos from the same group of videos of the same or similar recording remain in the same split to prevent data leakage. These splits are saved automatically for later testing.
- **Select frames for each video** When enabled, random temporal sampling selects new training frames on each video. Validation and test use fixed frame selections.
- **Apply training data augmentation** including random crops, horizontal flips, translations, and color jitter. Validation and testing uses resizing and normalization without random augmentation.
- **Create custom PyTorch Datasets and DataLoaders** for training and validation with padding and recorded lengths. The test DataLoader is created separately during evaluation.
- **Initialize the LRCN model** with the selected CNN backbone, recurrent layers, and normalization settings. The provided train.sh script uses ConvNeXt-Tiny, a BiLSTM, and LayerNorm.
- **Configure cross-entropy loss, AdamW, and a learning-rate scheduler**
- **Record training and validation loss and accuracy** in a local CSV file and in W&B when enabled.
- **Save the best model weights** whenever validation accuracy improves.
---

## Testing and Evaluation

### Step 2: Run Testing

Before testing, edit `test.sh`:

- **Set `--ckpt`** to the saved checkpoint you want to evaluate, such as `./runs/YOUR_RUN/models/best_model_wts.pt`.
- **Match the model settings** to the training run: CNN backbone, hidden size, number of recurrent layers, bidirectionality, LayerNorm, and number of classes. Keep the same `--fr_per_vid` value for a comparable evaluation.
- **Keep the saved run folder.** Evaluation loads `splits.npy` from the run folder containing the checkpoint's `models` directory. This reuses the original test split rather than creating a new one. The saved frame paths must still be accessible.

From the project directory, run:

```bash
bash test.sh
```

### Testing Script Overview

The testing script will:

- **Load the saved test split** associated with the checkpoint.
- **Select fixed, uniformly spaced frames** and apply resizing and normalization without random augmentation.
- **Create a DataLoader** that pads variable-length clips and records their original lengths.
- **Load the complete model weights** and run evaluation without updating them.
- **Print test accuracy and average test loss.** The loss is unsmoothed cross-entropy averaged over the evaluated videos.
- **Save the results locally** in `test_metrics_<checkpoint_name>.json` in the run folder, such as `test_metrics_best_model_wts.json`.
- **Log test metrics to W&B when enabled.** Matching online `tracking.json` metadata lets evaluation append results to the original training run. Without that metadata, it creates a separate evaluation run unless `--wandb_run_id` is supplied.

The terminal prints accuracy as a percentage.

---

## Unit Testing and Code Quality

### Pytest

`test_units.py` contains basic tests for frame sampling, group-disjoint splitting, padding and sequence lengths, model output shape, and evaluation loss and accuracy.

The tests use small example inputs and do not require the UCF50 dataset or downloaded pretrained weights. They check code functionality; trained-model accuracy is evaluated separately using `test.sh`.

Run the tests and save their output:

```bash
python -m pytest -v -s test_units.py > pytest_output.txt 2>&1
```

The `-v` option lists each test, and `-s` includes the explanatory print statements.

See [pytest_output.txt](pytest_output.txt) for the test results.

### Pylint

Pylint checks the Python files for issues such as unused imports, formatting problems, and overly complex functions. Its score and remaining findings are recorded in the report.

Run Pylint and save its output:

```bash
python -m pylint models.py preprocess.py run.py train.py test.py utils.py video_datasets.py tracking.py test_units.py > pylint_output.txt 2>&1
```

See [pylint_output.txt](pylint_output.txt) for the findings and score.

---

## Project Structure

| File or directory | Purpose |
|---|---|
| `preprocess.py` | Extract sampled video frames and save them as JPEG images. |
| `run.py` | Parse experiment settings and start training or evaluation. |
| `models.py` | Define the CNN and recurrent video-classification model. |
| `train.py` | Run training and validation, record metrics, and save the best weights. |
| `test.py` | Calculate test loss and accuracy and provide reporting helpers. |
| `test_units.py` | Run basic pytest checks for key pipeline functionality. |
| `video_datasets.py` | Load clips, sample frames, split by group, and pad batches. |
| `utils.py` | Provide frame-processing helpers, transforms, and DataLoaders. |
| `tracking.py` | Manage optional W&B logging and run metadata. |
| `train.sh` | Store training settings and launch training. |
| `test.sh` | Store evaluation settings and launch testing. |
| `requirements.txt` | List runtime Python dependencies. |
| `runs/` | Generated run folders containing checkpoints, saved splits, configuration, and metrics. |

---

## Customization and Hyperparameters

Edit the values directly inside `train.sh` and `test.sh` before running them. The project uses LRCN.

### Data Parameters

- `--frame_dir`: Directory containing extracted video frames, used when creating the training run.
- `--fr_per_vid`: Maximum number of frames used per video. This is separate from the number saved during preprocessing.
- `--random_temporal_sampling`: Select new training frames from temporal segments each time a video is loaded. Validation and testing remain deterministic.
- `--train_size` and `--test_size`: Requested training and test fractions. Validation uses the remaining fraction. Splitting keeps UCF50 group IDs separate, so actual sample percentages may differ from the requested fractions.
- `--batch_size`: Batch-size setting used by the DataLoaders. Training uses this value; validation and test loaders currently use twice this value.
- `--num_workers`: Number of CPU worker processes used to load data.

### Model Parameters

- `--cnn_backbone`: Choose `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, or `convnext_tiny`. The provided scripts use `convnext_tiny`.
- `--pretrained`: Use ImageNet CNN weights; enabled by default. To disable them, use `--pretrained false` or `--no-pretrained`.
- `--n_classes`: Number of output classes; use `50` for UCF50.
- `--rnn_hidden_size`: LSTM hidden size per direction. A bidirectional model with a hidden size of 100 produces 200 features per timestep.
- `--rnn_n_layers`: Number of LSTM layers.
- `--bidirectional`: Enable a BiLSTM that processes the sequence in both directions.
- `--rnn_layer_norm`: Apply LayerNorm to recurrent outputs before pooling.
- `--dropout`: Dropout probability before the final classifier.
- `--fine_tune_last_block`: Limit CNN fine-tuning to its final stage while keeping the recurrent head trainable. Omit this option for full CNN fine-tuning.

The model averages genuine recurrent timesteps after optional normalization. Padded timesteps are excluded from pooling.

### Training Parameters

- `--learning_rate`: Learning rate for the recurrent head, including its normalization and classifier parameters.
- `--cnn_learning_rate`: Separate CNN learning rate. If omitted, it uses `--learning_rate`.
- `--n_epochs`: Number of training epochs.
- `--weight_decay`: AdamW weight decay; `0.0` disables it.
- `--label_smoothing`: Training label smoothing; `0.0` disables it. Validation and test loss remain unsmoothed.

The best checkpoint is selected using validation accuracy. The learning-rate scheduler responds to validation loss.

### Evaluation and W&B Parameters

- `--ckpt`: Path to the model weights to evaluate.
- `--wandb`: Enable W&B logging.
- `--wandb_entity` and `--wandb_project`: Team and project receiving the results.
- `--wandb_name`: Optional display name for the run.
- `--wandb_mode`: Choose `online`, `offline`, or `disabled`.
- `--wandb_run_id`: Explicit existing W&B run ID for attaching evaluation results. This requires `--wandb`, evaluation mode, and online logging.

---

## Summary of Steps

- **Step 0: Dataset Preparation**  
  Download and extract UCF50, keeping videos organized by action class. Run `preprocess.py` to save frames in per-video folders.

- **Step 1: Run Training**  
  Configure the frame directory and training settings in `train.sh`, then run `bash train.sh`. Keep the resulting run folder, including its checkpoint and saved splits.

- **Step 2: Run Testing**  
  Set the checkpoint path and matching model settings in `test.sh`, then run `bash test.sh`. Review the test loss, test accuracy, saved results, and W&B metrics when enabled.
