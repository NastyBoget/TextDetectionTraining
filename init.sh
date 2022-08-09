rm -rf doctr
git clone https://github.com/mindee/doctr

cp fixed_files/train_pytorch.py doctr/references/detection/train_pytorch.py
cp fixed_files/detection.py doctr/doctr/datasets/detection.py

rm -rf data
python3 dataset/dataset_for_training.py https://at.ispras.ru/owncloud/index.php/s/8G9l1YXfhezhMPX/download
