#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Usage: ./upload.sh \"Your commit message\""
  exit 1
fi

# Remove any files that are currently tracked but should be ignored
echo "Cleaning tracked files that are ignored by .gitignore..."
git rm -r --cached . >/dev/null 2>&1
git add .

# Commit with the provided message
git commit -m "$1"

# Push to the current branch
git push
