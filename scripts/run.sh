#!/usr/bin/env bash

python3 -m finnessayscore.train \
	--jsons $DIR/ismi/ismi.json \
	--epochs 3
