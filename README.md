# Homework 4 (NV)

Implements HiFiGan Neural Vocoder model via PyTorch. [Wandb report in Russian](https://wandb.ai/dlishudi/NV/reports/HiFiGAN--VmlldzozMTk5OTIx?accessToken=i4k1y187bdobkabkp54s0lr5ylt7p5h03plua2rhjwt1i73bcmf9qp3ypq7w59na).

## Project Navigation

* hw_nv/ is main module of this project; it directly implements HiFiGan Neural Vocoder.
* requirements.txt lists all python requirements.
* setup.sh script installs needed requirements and downloads data and model checkpoint.
* train.py and synthesize.py scripts are used to train model and synthesize audio from mel spectrograms respectively.

## Requirements and Set Up
To install all requirements and download needed data and last checkpoint simply run:
```
./setup.sh
```
If you only need to install python requirements run:
```
pip install -r requirements.txt
```

## Model Training and Audio Synthesis
To train model run 
```
python ./train.py -c ./hw_nv/configs/config_V1.json
```
To synthesise audio run:
```
python3 ./synthesize.py -c ./hw_nv/configs/config_V1.json -w ./checkpoints/checkpoint_final.pth.tar
```
Mel spectrograms for synthesis should be placed in directory ./data/validation_mels.\
By default setup.sh downloads 3 mel spectrograms and corresponding source audio, and places them in this directory.

## References
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646), J. Kong, *et al*.
