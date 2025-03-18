# 🍅 Tomato Disease Classifier

A deep learning model to classify tomato plant diseases using PyTorch and ResNet18.

## Features

- Uses transfer learning (ResNet18)
- Modular structure
- Visual confusion matrix and classification report
- Easily extensible

## Dataset

Place your dataset like this:

```
data/
├── train/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   └── ...
├── val/
│   ├── Tomato___Bacterial_spot/
│   └── ...
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

## Evaluate the Model

```bash
python evaluate.py
```

## License

MIT
