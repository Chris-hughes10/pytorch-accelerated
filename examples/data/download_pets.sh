#!/bin/bash

wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -P pets
tar zxf pets/images.tar.gz -C pets
rm pets/images.tar.gz