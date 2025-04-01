#!/bin/bash


set -e

VAL_IMAGES_URL="http://es.esspeclist.kr:4341/val_images.tar"
VAL_IMAGES_TAR="val_images.tar"
VAL_IMAGES_DIR="val_images"

echo "📦 Start: Image Validation Set Downloading..."

wget $VAL_IMAGES_URL
tar xvf $VAL_IMAGES_TAR
rm -f $VAL_IMAGES_TAR
