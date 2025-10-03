#!/bin/bash
# Install system dependencies for OpenCV and MediaPipe
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0

# Make build.sh executable
chmod +x build.sh
