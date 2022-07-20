# Training doctr text detection model

This repository is for training the [doctr](https://mindee.github.io/doctr/) model on custom datasets.
For data labelling, the [ImageLabelerApp](https://github.com/dronperminov/ImageLabelerApp) may be used. 
See data_for_labelling.py for additional information how to make the input data for the ImageLabelerApp.

## How to train pretrained the doctr text detection model

### Create a new environment 
`python >= 3.6`

### Install needed packages using `pip`
```shell
pip install -r requirements.txt
```

### Clone `doctr` repository and download a custom labelled dataset
```shell
bash init.sh
```

### Run the training process of the pretrained `db_resnet50` model
```shell
bash train.sh
```

### Resume training the saved model
```shell
python3 doctr/references/detection/train_pytorch.py data/text_detection/train data/text_detection/val db_resnet50 --pretrained --epochs num_epochs --resume path_to_model --workers 1
```
