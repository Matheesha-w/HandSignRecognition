# Hand Sign Recognition

Real-time hand sign recognition using MediaPipe and RandomForest.
Supports 6 classes (0–5 fingers).

## Files

- `collect_imgs.py` – capture hand images
- `create_dataset.py` – extract hand landmarks
- `train_model.py` – train the model
- `inference.py` – real-time webcam prediction

## Usage

1. Collect images (optional if using sample data)
2. Create dataset: `python create_dataset.py`
3. Train model: `python train_model.py`
4. Run inference: `python inference.py`
