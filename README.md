# Audio Classification Project

A configurable audio classification pipeline for short audio clips (e.g. **car vs tram**).
The project is built around:

- **Hydra configuration**, 
- **Modular preprocessing**,
- **Exploration of audio features**, 
- **Flattening strategies** to apply features to classical ML models
- **Using classical machine-learning models**

---

## Dataset Structure

The dataset must be organized by class folders inside `dataset/`:

```
dataset/
├── car/
│ ├── car1.wav
│ └── ...
└── tram/
├── tram1.wav
└── ...
```

**Note:** The names of the files (like `car1.wav`) is not important; however, the directory (like `car/` and `tram/`) should be distinct because it is used to categorize samples into labels.


Each subfolder name is treated as a **class label**.

---

## Project Structure

```
├── dataset/ # Input dataset
├── result/
│ ├── metrics.csv # Evaluation metrics
│ └── model/
│ ├── svc.joblib
│ └── ...
├── src/
│ ├── data_loader.py # Dataset loading
│ ├── preprocessing.py # Padding/trimming, flattening
│ ├── features.py # Feature extraction: MFCC, STFT ...
│ ├── models.py # Model instantiation
│ ├── train.py # Model training
│ ├── evaluate.py # Evaluation + saving metrics
│ ├── utils.py # Shared helpers
│ └── run_pipeline.py # End-to-end pipeline
├── conf/
│ ├── config.yaml # Main Hydra configuration
│ └── models.yaml # Model definitions
├── main.py # Entry point
└── environment.yml
```

---

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate audio-classify
```

---

## Running the Pipeline

Make sure you are in the directory `audio_classification_project/` when running the pipeline

Run the full pipeline:

```bash 
python main.py 
```

This will:

* Load audio data
* Preprocess signals
* Extract multiple audio features
* Apply multiple flattening strategies
* Train multiple ML models
* Evaluate performance
* Save metrics and trained models

---

## Configuration & Command-Line Overrides

This project uses **Hydra**, which allows you to override any configuration value
directly from the command line **without modifying `config.yaml`**.

The default configuration is defined in `conf/config.yaml`, but you can change
parameters at runtime using:

```bash
python main.py <overrides>
```

---

## Common Overrides

### Enable / Disable Saving

```
# Disable model saving
python main.py save_model=false

# Disable metric saving
python main.py save_metrics=false
```

### Change Location

```
python main.py input_dir=/path/to/your/dataset

python main.py output.dir=/path/to/output
```

### Model Parameter Change

```
python main.py models.models.SVC.C=10.0 models.models.SVC.kernel=rbf
```

---

## Pipeline Overview

### 1. Data Loading

* Loads audio files from `dataset/<class>/*.wav`
* Uses `librosa.load`
* Sampling rate and extensions are configurable
* Outputs:

  * `X`: list of audio waveforms
  * `y`: list of class labels

---

### 2. Preprocessing

Each audio signal is **padded or trimmed** to a fixed duration using parameters from
`config.audio.duration`:

* Ensures consistent input length
* Required for stable feature extraction

---

### 3. Feature Extraction

For each audio signal, the following features are extracted:

| Feature          | Description                         |
| ---------------- | ----------------------------------- |
| MFCC             | Mel-frequency cepstral coefficients |
| STFT (magnitude) | Linear frequency spectrogram        |
| STFT (dB)        | Log-scaled STFT                     |
| Mel Spectrogram  | Perceptual frequency representation |
| CQT              | Constant-Q Transform                |

All features are returned as **3D arrays**:

```
(num_samples, frequency_bins, time_frames)
```

---

### 4. Flattening (Squeezing)

Classical ML models expect **2D input**:

```
(samples, features)
```

Each time–frequency feature is therefore flattened using configurable strategies.

#### Squeeze Types

| Type       | Description                                   |
| ---------- | --------------------------------------------- |
| `FLAT`     | Flatten frequency × time into a single vector |
| `AVG`      | Average values                                |
| `MAX`      | Max pooling                                   |
| `MEAN_STD` | Concatenate mean and standard deviation       |

#### Squeeze Axis

| Axis             | Meaning                                               |
| ---------------- | ----------------------------------------------------- |
| `OVER_TIME`      | Aggregate across time (keeps frequency structure)     |
| `OVER_FREQUENCY` | Aggregate across frequency (keeps temporal structure) |

Each **feature × squeeze type × axis** combination becomes a separate training input.

---

### 5. Train / Test Split

* Applied **after flattening**
* Stratified split
* Controlled by `config.test_proportion`

---

### 6. Models

Models are defined in `conf/models.yaml` and instantiated via Hydra.

Examples:

* Support Vector Classifier (SVC)
* Random Forest
* K-Nearest Neighbors
* Multi-Layer Perceptron

Each model is trained on **every flattened feature representation**.
It is very easy to add extra sklearn model or remove existing one in `cfg/models/models.yaml`

---

### 7. Training

* Uses classical ML (`model.fit`)

---

### 8. Evaluation

Each trained model is evaluated using:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)

---

### 9. Saving Outputs

If enabled in the config:

* Metrics are saved to:

  ```
  result/metrics.csv
  ```
* Trained models are saved to:

  ```
  result/model/<model_name>.joblib
  ```

All paths are configured in `cfg/config.yaml` and handled using `pathlib` and created automatically.