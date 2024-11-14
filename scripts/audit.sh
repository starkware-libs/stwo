#!/bin/bash

cargo install --locked cargo-deny
cargo deny check "$@"