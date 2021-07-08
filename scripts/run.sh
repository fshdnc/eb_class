#!/usr/bin/env bash

#source /home/ginter/venv-ptorch4/bin/activate
source /home/lhchan/venv-pl/bin/activate

DIR=/home/lhchan/eb_class
python3 $DIR/train.py \
	--jsons $DIR/ismi/ismi.json \
	--epochs 3

deactivate

