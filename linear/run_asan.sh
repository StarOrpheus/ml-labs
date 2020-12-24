#!/bin/sh
ninja -C _build_asan
./_build_asan/ml_lab2 test
