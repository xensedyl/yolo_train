#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolo

pyrcc5 -o resources.py resources.qrc
python roLabelImg.py

