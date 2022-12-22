#!/bin/bash

# install requirements
pip install -qq -r requirements.txt
# apt-get install -y -qq wget unzip

# prepare folders
mkdir data

# download LjSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
rm LJSpeech-1.1.tar.bz2 

# download validation melspectrograms
gdown --no-cookies https://drive.google.com/u/0/uc?id=19ZF7H5F0-5Eld5orEWWbH21ovnsmfS1a
tar -xvf validation_mels.tar >> /dev/null
mv validation_mels data/validation_mels

# download checkpoint
# mkdir ./fs2_saved
# gdown --no-cookies https://drive.google.com/file/d/19ZF7H5F0-5Eld5orEWWbH21ovnsmfS1a
# mv ./checkpoint_19999.pth.tar ./fs2_saved/checkpoint_19999.pth.tar