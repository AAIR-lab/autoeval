#!/bin/bash

EXTRA_OPTS=$1

apt update
apt install -y libgraphviz-dev \
  p7zip-full \
  graphviz \
  python3-pip \
  python3-dev \
  python-is-python3 \
  npm

cd prover9
mkdir -p bin
make all
cd ..

cd dependencies/reg2dfa
npm install
cd ../../

7z x -aoa dataset.zip

pip3 install --no-cache-dir $EXTRA_OPTS -r requirements.txt