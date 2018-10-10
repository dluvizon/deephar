#!/usr/bin/env python2

import os
import sys

try:
    fid = open(sys.argv[1], 'r')
except Exception as e:
    print (str(e) + '\n\nExpected the first argument to be the input file.')
    sys.exit()

import cv2

def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)

OUTDIR = 'images'
mkdir(OUTDIR)

def extract_frames(vidcap, prefix):
    if (prefix[-1] == '') or (prefix[-1] == '\n'):
        prefix = prefix[:-1]

    subject = int(prefix[5:7])
    print (prefix, subject)

    imgdir = OUTDIR + '/' + str(prefix)
    mkdir(imgdir)

    f = 0
    while True:
        success, image = vidcap.read()
        f += 1
        if not success:
            print ('End of file!')
            break

        if subject in [2, 3, 4, 10]: # Test
            continue
        elif subject in [1, 5, 6, 7, 8]: # Train
            if (f % 5) != 1:
                continue
        elif subject in [9, 11]: # Validation
            if (f % 64) != 1:
                continue
        else:
            raise Exception('Unexpected subject number ({})'.format(subject))

        print ('%s %d' % (prefix, f))
        fname = imgdir + '/%05d.jpg' % f
        cv2.imwrite(fname, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

try:
    while True:
        line = fid.readline()
        if line == '':
            break
        mp4, prefix = line.split(':')
        vidcap = cv2.VideoCapture(mp4)
        extract_frames(vidcap, prefix)
        vidcap.release()
except Exception as e:
    print (e)

