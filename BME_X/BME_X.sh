#!/bin/bash

# Ensure the script is being run with the correct number of arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 --bids_dir <bids_dir> --output_dir <output_dir> --subject <subject> --session <session>"
    exit 1
fi

# Parse arguments
BIDS_DIR=""
OUTPUT_DIR=""
SUBJECT=""
SESSION=""

# Loop through the arguments to assign values to variables
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bids_dir) BIDS_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --subject) SUBJECT="$2"; shift ;;
        --session) SESSION="$2"; shift ;;
        *) echo "Unknown option $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the provided arguments
echo "Running Python script with the following arguments:"
echo "--bids_dir $BIDS_DIR"
echo "--output_dir $OUTPUT_DIR"
echo "--subject $SUBJECT"
echo "--session $SESSION"

python3 BIDS_data.py --bids_dir "$BIDS_DIR" --output_dir "$OUTPUT_DIR" --subject "$SUBJECT" --session "$SESSION"
