import numpy as np

from .pose import get_valid_joints

from .io import WARNING
from .io import printcn

relsize_std = 1.5
square_std = True

class PoseBBox():
    def __init__(self, poses, relsize=relsize_std, square=square_std):
        self.poses = poses
        self.relsize = relsize
        self.square = square
        if len(poses.shape) == 4:
            self.num_frames = poses.shape[1]
        else:
            self.num_frames = None

    def __getitem__(self, key):
        p = self.poses[key]
        if isinstance(key, int):
            return self._get_bbox(p)
        if isinstance(key, slice):
            indices = key.indices(len(self))
            key = range(*indices)
        x = np.zeros((len(key),) + self.shape[1:])
        for i in range(len(key)):
            x[i,:] = self._get_bbox(p[i])
        return x

    def _get_bbox(self, p):
        if self.num_frames is None:
            return get_valid_bbox(p, relsize=self.relsize, square=self.square)
        else:
            b = np.zeros(self.shape[1:])
            for f in range(self.num_frames):
                b[f, :] = get_valid_bbox(p[f], self.relsize, self.square)
            return b

    def __len__(self):
        return len(self.poses)

    @property
    def shape(self):
        if self.num_frames is None:
            return (len(self), 4)
        else:
            return (len(self), self.num_frames, 4)

def get_valid_bbox(points, jprob=None, relsize=relsize_std, square=square_std):
    if jprob is None:
        v = get_valid_joints(points)
    else:
        v = np.squeeze(jprob > 0.5)

    if v.any():
        x = points[v==1, 0]
        y = points[v==1, 1]
    else:
        raise ValueError('get_valid_bbox: all points are invalid!')
        # printcn(WARNING, 'All points are invalid! ' + str(points))
        # x = np.array([0.5])
        # y = np.array([0.5])

    cx = (min(x) + max(x)) / 2.
    cy = (min(y) + max(y)) / 2.
    rw = (relsize * (max(x) - min(x))) / 2.
    rh = (relsize * (max(y) - min(y))) / 2.
    if square:
        rw = max(rw, rh)
        rh = max(rw, rh)

    return np.array([cx - rw, cy - rh, cx + rw, cy + rh])

def get_valid_bbox_array(pointarray, jprob=None, relsize=relsize_std,
        square=square_std):

    bboxes = np.zeros((len(pointarray), 4))
    v = None
    for i in range(len(pointarray)):
        if jprob is not None:
            v = jprob[i]
        bboxes[i, :] = get_valid_bbox(pointarray[i], jprob=v,
                relsize=relsize, square=square)

    return bboxes

def get_objpos_winsize(points, relsize=relsize_std, square=square_std):
    x = points[:, 0]
    y = points[:, 1]
    cx = (min(x) + max(x)) / 2.
    cy = (min(y) + max(y)) / 2.
    w = relsize * (max(x) - min(x))
    h = relsize * (max(y) - min(y))
    if square:
        w = max(w, h)
        h = max(w, h)

    return np.array([cx, cy]), (w, h)

def compute_grid_bboxes(frame_size, grid=(3, 2),
        relsize=relsize_std,
        square=square_std):

    bb_cnt = 0
    num_bb = 2 + grid[0]*grid[1]
    bboxes = np.zeros((num_bb, 4))

    def _smax(a, b):
        if square:
            return max(a, b), max(a, b)
        return a, b

    # Compute the first two bounding boxes as the full frame + relsize
    cx = frame_size[0] / 2
    cy = frame_size[1] / 2
    rw, rh = _smax(cx, cy)
    bboxes[bb_cnt, :] = np.array([cx-rw, cy-rh, cx+rw, cy+rh])
    bb_cnt += 1

    rw *= relsize
    rh *= relsize
    bboxes[bb_cnt, :] = np.array([cx-rw, cy-rh, cx+rw, cy+rh])
    bb_cnt += 1

    winrw = frame_size[0] / (grid[0]+1)
    winrh = frame_size[1] / (grid[1]+1)
    rw, rh = _smax(winrw, winrh)

    for j in range(1, grid[1]+1):
        for i in range(1, grid[0]+1):
            cx = i * winrw
            cy = j * winrh
            bboxes[bb_cnt, :] = np.array([cx-rw, cy-rh, cx+rw, cy+rh])
            bb_cnt += 1

    return bboxes

def bbox_to_objposwin(bbox):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    wx = bbox[2] - bbox[0]
    wy = bbox[3] - bbox[1]

    return np.array([cx, cy]), (wx, wy)

def objposwin_to_bbox(objpos, winsize):
    x1 = objpos[0] - winsize[0]/2
    y1 = objpos[1] - winsize[1]/2
    x2 = objpos[0] + winsize[0]/2
    y2 = objpos[1] + winsize[1]/2

    return np.array([x1, y1, x2, y2])

