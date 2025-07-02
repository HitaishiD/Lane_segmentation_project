#!/bin/bash

# Download the model weights
wget https://storage.googleapis.com/lane-segmentation/final_model.pth

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully"
else
    echo "Download failed. Exiting..."
    exit 1
fi

