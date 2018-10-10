import numpy as np

from deephar.utils import *


def get_clip_frame_index(sequence_size, subsample, num_frames,
        random_clip=False):

    # Assert that subsample is integer and positive
    assert (type(subsample) == int) and subsample > 0

    idx_coef = 1.
    while idx_coef*sequence_size < num_frames:
        idx_coef *= 1.5
    sequence_size *= idx_coef

    # Check if the given subsample value is feasible, otherwise, reduce
    # it to the maximum acceptable value.
    max_subsample = int(sequence_size / num_frames)
    if subsample > max_subsample:
        subsample = max_subsample

    vidminf = subsample * (num_frames - 1) + 1 # Video min num of frames
    maxs = sequence_size - vidminf # Maximum start
    if random_clip:
        start = np.random.randint(maxs + 1)
    else:
        start = int(maxs / 2)

    frames = list(range(start, start + vidminf, subsample))
    if idx_coef > 1:
        for i in range(len(frames)):
            frames[i] = int(frames[i] / idx_coef)

    return frames

