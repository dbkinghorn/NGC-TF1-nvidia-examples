
# ImageNet Dataset Build Scripts

This directory contains scripts for downloading and building the ImageNet
(ILSVRC 2012) dataset in a sharded TFRecord protobuf format suitable for
efficient training and validation.

The produced training set consists of 1024 TFRecord files, each containing
approximately 1251 JPEG images and associated metadata (including object
detection bounding boxes).

The produced validation set consists of 128 TFRecord files, each containing
approximately 391 JPEG images and associated metadata.

See nvidia-examples/cnn/nvutils/image_processing.py for an example of how to
load, parse, and preprocess these datasets as part of a training script.

## How to run the scripts

### Downloading and building a TFRecord dataset from scratch

1. Create an ImageNet account at http://image-net.org. You will need a user ID
   and the access key provided upon registration.
2. Run the script:

  `./download_and_preprocess_imagenet.sh <DATA_DIR>`

   where <DATA_DIR> is the path you would like output files to be written.

**Note:** If you already have the files ILSVRC2012_img_val.tar and
ILSVRC2012_img_train.tar, place them into <DATA_DIR>/raw-data/ before running
the script to avoid them being re-downloaded.

**Note:** By default, the script will skip any tfrecord files that already
exist in the output directory. If bad or partial files are present, please
delete them before running the script.

### Resizing an existing TFRecord dataset

1. Run the script, once for each subset:

   `./tfrecord_image_resizer.py -i <IN_DIR> -o <OUT_DIR> --subset_name train`

   `./tfrecord_image_resizer.py -i <IN_DIR> -o <OUT_DIR> --subset_name validation`

    where <IN_DIR> is the path to the existing TFRecord dataset, and <OUT_DIR>
    is the path to the new one.

## Image size and quality

By default, the scripts will shrink images so that they are at most 480 pixels
on the shortest side (maintaining aspect ratio) and have a JPEG quality factor
of 85. These parameters can be changed by modifying the flags at the bottom of
download_and_preprocess_imagenet.sh, or by passing --size=XX and --quality=XX
to the tfrecord_image_resizer.py script. A setting of size=0 will keep the
images at their original sizes.

In general, larger images and higher quality factors may improve training
accuracy, but this comes at the cost of a larger dataset and slower input
pipeline.

## Troubleshooting

The download_and_preprocess_imagenet.sh script will attempt to use existing
files if they are present (e.g., from an incomplete prior run). In some cases
this can lead to a incomplete dataset generation if a tarball was not fully
uncompressed. If you suspect such an issue, delete the subdirectories in
<DATA_DIR>/raw-data/ before re-running the script.
