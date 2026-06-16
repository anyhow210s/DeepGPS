# Photo Geolocation Prediction using Deep Learning

**English:** This README explains the project goal, dataset preparation, model training, evaluation, prediction, and explainability workflow.

**한국어:** 이 README는 프로젝트 목표, 데이터셋 준비, 모델 학습, 평가, 예측, 설명 가능성 분석 과정을 안내합니다.

This project trains a deep learning image classifier to predict the location category of a photo without using EXIF GPS metadata. It is designed as a deep learning course assignment using PyTorch transfer learning with a pretrained ResNet50 convolutional neural network (CNN).

The model learns visual clues such as landscape, architecture, vegetation, road patterns, and color distributions from a local image dataset that has already been manually organized by location. It does not call any external AI API or online photo service.

## Why This Is Deep Learning

ResNet50 is a deep convolutional neural network pretrained on ImageNet. Its convolutional layers extract hierarchical visual features, while a newly trained classification head maps those features to location classes. Training happens in two stages:

1. Freeze the pretrained ResNet50 backbone and train only the final classifier head.
2. Optionally unfreeze the last ResNet block (`layer4`) and fine-tune it using a smaller learning rate.

## Project Structure

```text
Photo Geolocation Prediction using Deep Learning/
├── data/
│   ├── raw/
│   │   ├── Location_A/
│   │   ├── Location_B/
│   │   ├── Location_C/
│   │   └── ...
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_training_results.ipynb
├── outputs/
│   ├── models/
│   ├── figures/
│   └── reports/
├── src/
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── gradcam.py
│   └── app.py
├── README.md
└── requirements.txt
```

## Dataset

This project assumes that the image dataset has already been manually classified into one folder per location. The model does not use GPS metadata, EXIF coordinates, or any online photo service connection.

Use this expected folder structure:

```text
data/raw/
  Location_A/
  Location_B/
  Location_C/
  ...
```

Each folder name becomes one location class. For example, if the folder is named `Yosemite`, the model treats `Yosemite` as one prediction category.

Each class folder may contain JPG, JPEG, PNG, HEIC, or HEIF images. JSON files, videos, and unsupported files are ignored. Corrupted images are skipped. The preparation script converts validated images into standard RGB JPEG files in `data/processed/`, leaving the original raw images unchanged.

Avoid using near-duplicate burst photos across location classes, and try to collect a balanced number of photos for each location.

## Setup on MacBook Apple Silicon

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch automatically uses Apple Silicon Metal Performance Shaders (`mps`) when available. If MPS is unavailable, the scripts fall back to CPU.

## Quick Run Commands

Use the commands below from the project root folder.

```bash
cd /Users/koscom/Documents/DeepGPS
```

If you are using the Anaconda Python environment on this Mac:

```bash
/opt/anaconda3/bin/python -m streamlit run src/app.py
```

If you are using a virtual environment:

```bash
source .venv/bin/activate
streamlit run src/app.py
```

The Streamlit UI is the easiest way to demonstrate the project. It lets you select Korean or English, enter the local photo folder path, prepare the dataset, start training, run evaluation, upload a test photo, and view Grad-CAM results.

For command-line execution, run the project in this order:

```bash
python src/prepare_dataset.py --overwrite
python src/train.py
python src/evaluate.py
python src/predict.py --image path/to/photo.jpg
python src/gradcam.py --image path/to/photo.jpg
```

한국어 요약:

```bash
# 1. 프로젝트 폴더로 이동
cd /Users/koscom/Documents/DeepGPS

# 2. Streamlit UI 실행
/opt/anaconda3/bin/python -m streamlit run src/app.py

# 3. 터미널에서 직접 실행하는 경우
python src/prepare_dataset.py --overwrite
python src/train.py
python src/evaluate.py
python src/predict.py --image path/to/photo.jpg
python src/gradcam.py --image path/to/photo.jpg
```

## Prepare the Dataset

The script validates source images, prints class counts, and creates a reproducible 70% training, 15% validation, and 15% test split.
Each class needs at least seven valid images so every split is non-empty, though substantially more images are recommended.

```bash
python src/prepare_dataset.py
```

To recreate an existing processed split:

```bash
python src/prepare_dataset.py --overwrite
```

The images are resized to 224 x 224 during model loading. Training augmentation includes:

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `ColorJitter`
- `RandomRotation`
- ImageNet mean and standard deviation normalization

## Train the Model

```bash
python src/train.py
```

Useful options:

```bash
python src/train.py \
  --batch-size 32 \
  --head-epochs 5 \
  --fine-tune-epochs 5 \
  --learning-rate 0.001 \
  --fine-tune-learning-rate 0.0001
```

Training uses `CrossEntropyLoss` and the Adam optimizer. The best checkpoint is selected by validation accuracy and saved to:

```text
outputs/models/best_resnet50.pth
```

The script also saves:

- `outputs/reports/training_history.csv`
- `outputs/reports/class_names.json`
- `outputs/figures/training_history.png`

## Evaluate the Model

```bash
python src/evaluate.py
```

Evaluation uses the held-out test split and reports test accuracy, per-class precision, recall, F1-score, and a confusion matrix.

Saved outputs:

- `outputs/reports/evaluation_summary.txt`
- `outputs/reports/classification_report.csv`
- `outputs/figures/confusion_matrix.png`

## Predict a New Photo

```bash
python src/predict.py --image path/to/photo.jpg
```

The script prints the top three predicted locations with probabilities. Use `--top-k` to change the number of predictions.

## Generate a Grad-CAM Explanation

```bash
python src/gradcam.py --image path/to/photo.jpg
```

Grad-CAM uses gradients from the final ResNet50 convolution block to highlight image regions that influenced the predicted class. The resulting image is saved under `outputs/figures/`.

## Train and View Results in the Streamlit UI

Start the local Streamlit dashboard:

```bash
streamlit run src/app.py
```

The UI opens in a web browser and provides three tabs:

- **Data & Training:** Select the local folder containing location subfolders, prepare the dataset, train ResNet50, and optionally run test evaluation automatically.
- **Photo Prediction:** Upload a JPG, PNG, or HEIC photo, view the top three predicted locations with probabilities, and inspect a Grad-CAM explanation.
- **Training Results:** View saved training and validation curves, the classification report, the confusion matrix, and the evaluation summary.

On macOS, click `Choose Photo Folder` in the dashboard to select the manually classified image folder with a native folder dialog. The selected path is shown below the button.

The selected folder must contain one subfolder per location class:

```text
data/raw/
  Location_A/
  Location_B/
  Location_C/
  ...
```

When training is launched from the UI, it runs the same local scripts:

1. `src/prepare_dataset.py`
2. `src/train.py`
3. `src/evaluate.py` if evaluation is enabled

The UI uses only local files and the locally trained model at `outputs/models/best_resnet50.pth`. It does not send photos to an external API.

## Results Interpretation

- Compare training and validation loss to identify overfitting.
- Compare training and validation accuracy to assess generalization.
- Use the confusion matrix to find location pairs that the model commonly confuses.
- Use per-class precision, recall, and F1-score to identify weak or underrepresented classes.
- Inspect Grad-CAM images to determine whether the model focuses on meaningful landmarks and scenery or on accidental shortcuts.

A high validation accuracy does not guarantee real-world geolocation ability. The model predicts only among the classes used during training.

## Limitations

- The model cannot predict locations outside the trained class list.
- Personal photo collections may contain class imbalance, duplicates, seasonal bias, or photographer-specific bias.
- Visually similar cities and landscapes can be difficult to distinguish.
- Indoor photos may provide little useful geographic information.
- The model can learn shortcuts such as recurring people, vehicles, or camera artifacts.
- This project intentionally does not use GPS metadata.

## Future Work

- Compare ResNet50 with EfficientNetB0 or a Vision Transformer.
- Add stronger duplicate detection and class balancing.
- Use hierarchical labels such as country, region, and city.
- Add confidence calibration and an "unknown location" rejection threshold.
- Add an optional EXIF GPS auto-suggestion tool only for helping organize raw photos before training. EXIF GPS must remain excluded from model inputs.

## Notebooks

`01_data_exploration.ipynb` summarizes class counts and displays sample images. `02_training_results.ipynb` visualizes training history, evaluation metrics, and the saved confusion matrix after the scripts have been run.
