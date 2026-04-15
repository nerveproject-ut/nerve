#!/bin/bash

# first argument is the input file (.raw), second one is the output file (.hdf5)
# checks are done by python script.
input_file=$1
output_file=$2

# Now let's use the python inside project virtual env
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
exec $SCRIPT_DIR/../../../venv/bin/python $SCRIPT_DIR/raw_to_hdf5.py -i $input_file -o $output_file