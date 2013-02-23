#!/bin/bash

git submodule init
git submodule update
CWD=$PWD
cd sift/im
git checkout master
cd "$CWD"
git submodule sync
git submodule foreach git pull

