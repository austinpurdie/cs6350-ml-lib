#!/bin/bash

pip install numpy
pip install pandas
pip install random

echo "Will sleep five minutes after execution so you can view the results..."

echo "Running hw4-prob2.py..."
python SVM/hw4-prob2.py

sleep 300