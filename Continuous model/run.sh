#!/bin/bash

set -e


wd=$(pwd)
mkdir  $wd/output



python3 train_NF.py\
  --data-dir data/ \
  --output-dir $wd/output/
