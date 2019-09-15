#!/bin/bash

proj="/home/ikmlab/p76061124/ElaAdmin-master"
cd $proj
source env/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py &
