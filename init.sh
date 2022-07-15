git clone https://github.com/mindee/doctr

cp train_pytorch.py doctr/references/detection/train_pytorch.py
cp detection.py doctr/doctr/datasets/detection.py

python3 dataset_for_training.py
