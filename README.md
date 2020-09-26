# medical_mask_detection

## Dataset Referance
The original dataset is prepared by Prajna Bhandary and available at [github.com/prajnasb](https://github.com/prajnasb/observations/tree/master/experiements/data)

Then place it under the "dataset" folder

## Requirements
```
pip install imutils
pip install opencv-python
```
## If you're first time running
Firstly
```shell
python data_preprocessing.py
```
later
```shell
python model_train.py
```
the above is enough to do once.

## Run
```shell
python mask_detection.py
```

## Mask Detection
![screenshot of conversion](https://github.com/fbasatemur/medical_mask_detection/blob/master/Figure_Masked.jpg)

## Accuracy
![screenshot of conversion](https://github.com/fbasatemur/medical_mask_detection/blob/master/Figure_Accuracy.png)

