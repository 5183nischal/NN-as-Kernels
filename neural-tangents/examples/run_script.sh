#!/bin/bash

echo starting to run
export PATH = "ems/elsc-labs/sompolinsky-h/nischal.mainali/anaconda3/bin:PATH"
export PYTHONDIR = "ems/elsc-labs/sompolinsky-h/nischal.mainali/anaconda3/bin:PATH"

PWD = $(ped)
echo $PATH
python nn+gp01.py