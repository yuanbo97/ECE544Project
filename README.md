# ECE544Project


## Objective

Course project: node detection in plant root picture.

## Required library:

conda

pytorch

matplotlib

python 3.8+


## Prerequisite:

run preprocess.py to parse the images into patches and generate corresponding label

## How to train:

run train.py

## How to test:

modify the variable "now" inside test.py, so that it reflects the latest model timestamp. It has the following format: "MM_DD_hh_mm" e.g. "09_12_14_57" means Sep 12 14:57

then run 

python test.py --pic 0

or run

sh getFinalInference.sh


## Where to final evaluation result:

inside ./compare folder
