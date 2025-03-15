#!/bin/bash

# Download the dataset
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully. Now unzipping..."
    # Unzip the downloaded file
    unzip data_semantics.zip
else
    echo "Download failed. Exiting..."
    exit 1
fi
