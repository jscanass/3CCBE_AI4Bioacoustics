#!/bin/bash

# Install additional dependencies
uv pip install "wavio>=0.0.9"
uv pip install "batdetect2==1.0.8"

# Download data
gdown 1lBvKnORB-1wUcZtac59O_fwS3JRanM-a

# Unzip data
unzip 3ccbe_data.zip

# Download utility modules
wget https://github.com/jscanass/3CCBE_AI4Bioacoustics/raw/refs/heads/main/evaluation_utils.py
wget https://github.com/jscanass/3CCBE_AI4Bioacoustics/raw/refs/heads/main/plotting_utils.py
wget https://github.com/jscanass/3CCBE_AI4Bioacoustics/raw/refs/heads/main/audio_utils.py
wget https://github.com/jscanass/3CCBE_AI4Bioacoustics/raw/refs/heads/main/utils.py

