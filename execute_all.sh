#!/bin/bash

# Directory containing the folders
BASE_DIR="$(dirname "$0")"

# Loop over all folders in the base directory
for dir in "$BASE_DIR"/*/; do
  # Check if it is a directory
  if [ -d "$dir" ]; then
    # Change to the directory
    pushd "$dir"

    # Remove all .csv in folder
    rm -rf *.csv
    
    # Find all Python files in the directory
    for PYTHON_FILE in *.py; do
      # Check if the Python file exists
      if [ -f "$PYTHON_FILE" ]; then
        # Execute the Python file
        python3 "$PYTHON_FILE"
      fi
    done
    
    # Check if the output_ParaView folder exists
    if [ -d "output_ParaView" ]; then
      # Remove all content in the output_ParaView folder
      rm -rf output_ParaView/*
    else
      # Create the output_ParaView folder
      mkdir output_ParaView
    fi
    
    # Move all .xdmf and .h5 files into the output_ParaView folder
    mv *.xdmf *.h5 output_ParaView/ 2>/dev/null
    
    # Change back to the base directory
    popd
  fi
done