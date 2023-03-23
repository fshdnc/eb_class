#!/usr/bin/env bash

for lr in 1e-5 2e-5 3e-5; do
    for ga in 2 4 6; do
	for i in 1 2 3; do
	    #sbatch scripts/puhti.sh "$lr" "$ga"
	    echo "$lr" "$ga"
	done
    done
done
