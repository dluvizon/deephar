#!/usr/bin/env python2

import os
import sys
import glob
import threading

import numpy as np
from PIL import Image

import cv2

resize_ratio = 1/2.


def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)


def extract_video(video_file, output_dir, frames, jpeg_quality=95):

    try:
        vidcap = cv2.VideoCapture(video_file)
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        sys.stderr.write('Error loading file "{}"\n'.format(video_file))
        return

    mkdir(output_dir)
    sys.stdout.write('Extracting video "{}"\n'.format(video_file))
    sys.stdout.flush()

    f = 0
    while True:
        success, image = vidcap.read()
        f += 1
        if not success:
            break
        if f not in frames:
            continue # Skip if not in the frame list

        ncols, nrows, rgb = image.shape
        dsize = (int(resize_ratio*nrows), int(resize_ratio*ncols))
        image = cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC)

        frame_file = os.path.join(output_dir, '%05d.jpg' % f)
        cv2.imwrite(frame_file, image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    vidcap.release()


def extract_resize_videos(input_path, output_path, annot_path, subjects,
        num_C=3, num_P=40, num_R=2, num_A=60):

    if os.path.isdir(input_path) is False:
        raise Exception('Could not find the "{}" folder!'.format(input_path))

    mkdir(output_path)

    for s in subjects:
        for c in range(1,num_C+1):
            for p in range(1,num_P+1):
                for r in range(1,num_R+1):
                    for a in range(1,num_A+1):

                        sequence_id = 'S%03dC%03dP%03dR%03dA%03d' \
                                % (s, c, p, r, a)

                        video_file = os.path.join(input_path,
                                sequence_id + '_rgb.avi')

                        annot_file = os.path.join(annot_path,
                                sequence_id + '.npy')
                        if not os.path.isfile(annot_file):
                            continue # Ignore missing annotation files

                        annot = np.load(annot_file)
                        # frames = annot[1::2,0] # Divide video fps by 2
                        frames = annot[:,0]

                        output_dir = os.path.join(output_path, sequence_id)
                        extract_video(video_file, output_dir, frames)


if __name__ == "__main__":
    try:
        assert len(sys.argv) == 2, \
                'Expected 1 parameter, got {}'.format(sys.argv)
        setlist = [int(sys.argv[-1])]
        extract_resize_videos(
                'nturgb+d_rgb', 'images-small', 'nturgb+d_numpy', setlist)
    except Exception as e:
        print (e)

