#!/bin/bash

# Install additional dependencies
uv pip install "wavio>=0.0.9"
uv pip install "batdetect2==1.0.8"

# Download data
gdown 1UWPoP6J9bh9l7Q-cRYsd2I6rrQGnjMUJ

# Unzip data
unzip acoustic_data.zip

# Download utility modules
wget https://github.com/mbsantiago/AI-Intervene-Training-Material/raw/refs/heads/main/BioacousticsAI/evaluation_utils.py
wget https://github.com/mbsantiago/AI-Intervene-Training-Material/raw/refs/heads/main/BioacousticsAI/plotting_utils.py
wget https://github.com/mbsantiago/AI-Intervene-Training-Material/raw/refs/heads/main/BioacousticsAI/audio_utils.py
