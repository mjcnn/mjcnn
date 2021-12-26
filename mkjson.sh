#!/bin/bash
cat vgg16_weights.txt| egrep -v '(DATATYPE|DATASPACE|STRSIZE|STRPAD|CSET|CTYPE)' | sed 's/^.*)://g;s/DATA /"DATA": /g;s/DATASET \(.*\) {/"DATASET": { \1\,/g;s/GROUP \(.*\) {/"GROUP": { \1\,/g;s/ATTRIBUTE \(.*\) {/"ATTRIBUTE": { \1\,/g;s/HDF5 \(.*\) {/"HDF5": { \1\,/g' > vgg16_weights.json

