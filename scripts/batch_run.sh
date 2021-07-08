#!/usr/bin/env bash

RESULT_FILE=$1

SCRIPT=/home/lhchan/eb_class/scripts/run.sh
# run
for i in $(seq 20);do
    echo -ne "$i\t"
    bash $SCRIPT 1>> $RESULT_FILE 2>/dev/null
done

echo -e "RESULT FILE\t$RESULT_FILE"
# Training accuracy
echo "Training accuracy"
cat $RESULT_FILE | grep -A 4 'Training set' | grep Acc | grep -Po '\d.*$' | python3 /home/lhchan/eb_class/scripts/average.py
echo "Number of predicted classes (Training)"
cat $RESULT_FILE | grep -A 4 'Training set' | grep Predicted | grep -Po '\d.*$' | python3 /home/lhchan/eb_class/scripts/average.py

# Validation accuracy
echo "Validation accuracy"
cat $RESULT_FILE | grep -A 4 'Validation set' | grep Acc | grep -Po '\d.*$' | python3 /home/lhchan/eb_class/scripts/average.py
echo "Number of predicted classes (Validation)"
cat $RESULT_FILE | grep -A 4 'Validation set' | grep Predicted | grep -Po '\d.*$' | python3 /home/lhchan/eb_class/scripts/average.py
