#!/bin/bash

# Project directory is the current working directory
PROJECT_DIR="$PWD"

# Generate requirements.txt using pipreqs
pipreqs "$PROJECT_DIR"

# Check if any changes were made to requirements.txt
if [[ -n $(git -C "$PROJECT_DIR" status --porcelain requirements.txt) ]]; then
    # Add the generated dependencies file to Git
    git -C "$PROJECT_DIR" add requirements.txt
    git -C "$PROJECT_DIR" commit -m "Update dependencies"
    git -C "$PROJECT_DIR" push origin main  # Adjust if you are using a different branch
else
    echo "No changes to requirements.txt. Dependencies are already up to date."
fi