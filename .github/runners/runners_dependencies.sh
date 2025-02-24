#!/bin/bash

set -e

# Install esential linux dependencies.
sudo apt update
sudo apt install -y \
    build-essential \
    curl \
    git
