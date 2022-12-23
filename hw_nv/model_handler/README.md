# Homework 4 (NV)

## Model handlers submodule

This submodule implements **NVSynthesizer** and **NVTrainer** classes, which wrap around HiFiGan models.\
* NVSynthesizer is used for audio synthesis from melspectrograms and is used in *synthesize.py*.\
* NCTrainer inherits from it and also implements model training; it is used in *train.py*.
* utils.py contains functions to apply and remove weight/spectral normalization and to initialize weights from normal distribution. 
