#!/bin/zsh

#PBS -V
#PBS -d "/lustre/aoc/users/bsvoboda/temp/keystone_nestfit/"
#PBS -L tasks=1:lprocs=16:memory=32gb
#PBS -q batch

source /users/bsvoboda/.zshrc

python3 -m analyze --run-nested


