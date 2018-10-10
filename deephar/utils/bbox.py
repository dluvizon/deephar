import numpy as np

from .pose import get_valid_joints

from .io import WARNING
from .io import printcn
from .io import warning

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


logkey_warn = set()
def get_gt_bbox(pose, visible, image_size, scale=1.0, logkey=None):
    assert len(pose.shape) == 3 and pose.shape[-1] >= 2, \
            'Invalid pose shape ({})'.format(pose.shape) \
            + ', expected (num_frames, num_joints, dim) vector'
    assert len(pose) == len(visible), \
            'pose and visible should have the same langth'

    if len(pose) == 1:
        idx = [0]
    else:
        idx = [0, int(len(pose)/2 + 0.5), len(pose)-1]

    clip_bbox = np.array([np.inf, np.inf, -np.inf, -np.inf])

    for i in idx:
        temp = pose[i, visible[i] >= 0.5]
        if len(temp) == 0:
            temp = pose[i, pose[i] > 0]

        if len(temp) > 0:
            b = get_valid_bbox(temp, relsize=1.5*scale)

            clip_bbox[0] = min(b[0], clip_bbox[0])
            clip_bbox[1] = min(b[1], clip_bbox[1])
            clip_bbox[2] = max(b[2], clip_bbox[2])
            clip_bbox[3] = max(b[3], clip_bbox[3])
        else:
            if logkey not in logkey_warn:
                warning('No ground-truth bounding box, ' \
                        'using full image (key {})!'.format(logkey))
            logkey_warn.add(logkey)

            clip_bbox[0] = min(0, clip_bbox[0])
            clip_bbox[1] = min(0, clip_bbox[1])
            clip_bbox[2] = max(image_size[0], clip_bbox[2])
            clip_bbox[3] = max(image_size[1], clip_bbox[3])

    return clip_bbox


def get_crop_params(rootj, imgsize, f, scale):
    assert len(rootj.shape) == 2 and rootj.shape[-1] == 3, 'Invalid rootj ' \
            + 'shape ({}), expected (n, 3) vector'.format(rootj.shape)

    if len(rootj) == 1:
        idx = [0]
    else:
        idx = [0, int(len(rootj)/2 + 0.5), len(rootj)-1]

    x1 = y1 = np.inf
    x2 = y2 = -np.inf
    zrange = np.array([np.inf, -np.inf])
    for i in idx:
        objpos = np.array([rootj[0, 0], rootj[0, 1] + scale])
        d = rootj[0, 2]
        winsize = (2.25*scale)*max(imgsize[0]*f[0, 0]/d, imgsize[1]*f[0, 1]/d)
        bo = objposwin_to_bbox(objpos, (winsize, winsize))
        x1 = min(x1, bo[0])
        y1 = min(y1, bo[1])
        x2 = max(x2, bo[2])
        y2 = max(y2, bo[3])
        zrange[0] = min(zrange[0], d - scale*1000.)
        zrange[1] = max(zrange[1], d + scale*1000.)

    objpos, winsize = bbox_to_objposwin([x1, y1, x2, y2])

    return objpos, winsize, zrange

