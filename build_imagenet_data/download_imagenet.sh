#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script to download ImageNet Challenge 2012 training and validation data set.
#
# Downloads and decompresses raw images and bounding boxes.
#
# **IMPORTANT**
# To download the raw images, the user must create an account with image-net.org
# and generate a username and access_key. The latter two are required for
# downloading the raw images.
#
# usage:
#  ./download_imagenet.sh [dir name] [synsets file]
set -e

if [ "x$IMAGENET_ACCESS_KEY" == x -o "x$IMAGENET_USERNAME" == x ]; then
  cat <<END
In order to download the imagenet data, you have to create an account with
image-net.org. This will get you a username and an access key. You can set the
IMAGENET_USERNAME and IMAGENET_ACCESS_KEY environment variables, or you can
enter the credentials here.
END
  read -p "Username: " IMAGENET_USERNAME
  read -p "Access key: " IMAGENET_ACCESS_KEY
fi

OUTDIR="${1:-./imagenet-data}"
SYNSETS_FILE="${2:-./synsets.txt}"

# Make sure axel is installed
command -v axel >/dev/null 2>&1 ||
    { apt-get update && apt-get install -y --no-install-recommends axel; }

echo "Saving downloaded files to $OUTDIR"
mkdir -p "${OUTDIR}"
INITIAL_DIR=$(pwd)
cd "${OUTDIR}"

axel_safe() {
  # Don't download if file already exists and there is no state file
  if [ ! -f $2 ] || [ -f $2.st ]; then
    axel -a --num-connections=4 $1 -o $2
  else
    echo "Using existing file $2"
  fi
}

BBOX_FILENAME="ILSVRC2012_bbox_train_v2.tar.gz"
BASE_URL="http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112"

# Download and process all of the ImageNet bounding boxes.
# See here for details: http://www.image-net.org/download-bboxes
BOUNDING_BOX_ANNOTATIONS="${BASE_URL}/${BBOX_FILENAME}"
BBOX_TAR_BALL="${BBOX_FILENAME}"
echo "Downloading bounding box annotations."
axel_safe "${BOUNDING_BOX_ANNOTATIONS}" "${BBOX_TAR_BALL}"
BBOX_DIR="bounding_boxes"
if [ ! -d "${BBOX_DIR}" ]; then
  mkdir "${BBOX_DIR}"
  echo "Uncompressing bounding box annotations"
  tar xzf "${BBOX_TAR_BALL}" -C "${BBOX_DIR}" --checkpoint=.1000
  echo ""
else
  echo "Using existing files in ${BBOX_DIR}"
fi
echo "Counting bounding box annotations"
LABELS_ANNOTATED="${BBOX_DIR}/*"
NUM_XML=$(ls -1 ${LABELS_ANNOTATED} | wc -l)
echo "Identified ${NUM_XML} bounding box annotations."
# Download and uncompress all images from the ImageNet 2012 validation dataset.
VALIDATION_TARBALL="ILSVRC2012_img_val.tar"
echo "Downloading ${VALIDATION_TARBALL}"
axel_safe "${BASE_URL}/${VALIDATION_TARBALL}" "${VALIDATION_TARBALL}"
echo "Uncompressing ${VALIDATION_TARBALL}"
VALIDATION_PATH="validation"
if [ ! -d "${VALIDATION_PATH}" ]; then
  mkdir "${VALIDATION_PATH}"
  tar xf "${VALIDATION_TARBALL}" -C "${VALIDATION_PATH}" --checkpoint=.10000
  echo ""
else
  echo "Using existing files in ${VALIDATION_PATH}"
fi

# Download all images from the ImageNet 2012 train dataset.
TRAIN_TARBALL="ILSVRC2012_img_train.tar"
echo "Downloading ${TRAIN_TARBALL}"
axel_safe "${BASE_URL}/${TRAIN_TARBALL}" "${TRAIN_TARBALL}"
TRAIN_PATH="train"
mkdir -p "${TRAIN_PATH}"

# Un-compress the individual tar-files within the train tar-file.
echo "Uncompressing individual train tar-balls in the training data."

while read SYNSET; do
  # Check if directory already exists or was not completely processed
  if [ ! -d "${TRAIN_PATH}/${SYNSET}" ] || [ -f "${SYNSET}.tar" ]; then
    mkdir -p "${TRAIN_PATH}/${SYNSET}"
    # Uncompress into the directory.
    echo "Uncompressing train data to ${TRAIN_PATH}/${SYNSET}"
    tar xf "${TRAIN_TARBALL}" "${SYNSET}.tar"
    tar xf "${SYNSET}.tar" -C "${TRAIN_PATH}/${SYNSET}/"
    rm -f "${SYNSET}.tar"
  else
    echo "Using existing files in ${TRAIN_PATH}/${SYNSET}"
  fi
done < "${INITIAL_DIR}/${SYNSETS_FILE}"
