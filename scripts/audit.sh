#!/bin/bash

cargo install --version 0.16.1 cargo-deny
cargo deny check "$@"