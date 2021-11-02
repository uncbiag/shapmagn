#!/bin/bash

python dirlab_processing.py $1
python prepare_training_data.py $2 $3