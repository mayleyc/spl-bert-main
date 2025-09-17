#!/usr/bin/env bash

# List of configuration files
config_files=("src/models/GACapsHTC/configs/wos.json" "src/models/GACapsHTC/configs/bgc.json" "src/models/GACapsHTC/configs/bugs.json" "src/models/GACapsHTC/configs/rcv1.json" "src/models/GACapsHTC/configs/amazon.json")

# Loop over the configuration files
for CONFIG_FILE in "${config_files[@]}"
do
   echo "Running ...${CONFIG_FILE}"
   python src/models/GACapsHTC/run.py "${CONFIG_FILE}"
done