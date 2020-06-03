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
converted annotation files and some useful scripts for practical purposes. Some datasets, like Human3.6M, PennAction and NTU are composed of video files. In this software, we assume that all video frames are already extract. For this, you can use standard tools like `ffmpeg` or `opencv`.

### MPII

The **MPII Human Pose Dataset** is available in this [ [link](human-pose.mpi-inf.mpg.de) ].

Images from MPII should be manually downloaded and placed
at `datasets/MPII/images`.

### Human3.6M

The **Human3.6M** dataset is available in this [ [link](vision.imar.ro/human3.6m) ].

Videos from Human3.6M should be manually downloaded and placed
in `datasets/Human3.6M/S*`, e.g. S1, S2, S3, etc. for each subject.
After that, extract videos with:
```
  cd datasets/Human3.6M
  python2 vid2jpeg.py vid2jpeg.txt
```
Python2 is used here due to the dependency on cv2 package.

### PennAction

The **PennAction** dataset is available in this [ [link](http://dreamdragon.github.io/PennAction) ].

Video frames from PennAction should be manually downloaded and extracted
in `datasets/PennAction/frames`. The pose annotations and predicted bounding
boxes will be automatically downloaded by this software.

### NTU

The **NTU RGB+D** dataset is available in this [ [link](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) ].

Video frames from NTU should be also manually extracted.
A Python [script](datasets/NTU/extract-resize-videos.py) is provided to help in
this task. Python 2 is required.

Additional pose annotation is provided for NTU, which is used to train the pose
estimation part for this dataset. It is different from the original Kinect
poses, since it is a composition of 2D coordinates in RGB frames plus depth.
This additional annotation can be downloaded
[here](https://drive.google.com/open?id=1eTJPb8q2XCRK8NEC4h17p17JW2DDNwjG)
(2GB from Google Drive).

