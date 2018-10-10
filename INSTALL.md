# Installation guide

This file is a simplified guide to install (set up) the software and the
datasets.

## Limitations

Note that python3 is **required** and is used unless specified.

On Keras and on TensorFlow, only data format 'channels_last' is supported.

## Dependencies

Install required python packages before you continue:
```
  pip3 install -r requirements.txt
```

## Public datasets

We do not provide public datasets within this software. We only provide
converted annotation files and some useful scripts for practical purposes.

Images from MPII should be manually downloaded and placed
at `datasets/MPII/images`.

Videos from Human3.6M should be manually downloaded and placed
at `datasets/Human3.6M/S*`, e.g. S1, S2, S3, etc. for each subject.
After that, extract videos with:
```
  cd datasets/Human3.6M
  python2 vid2jpeg.py vid2jpeg.txt
```
Python2 is used here due to the dependency on cv2 package.

