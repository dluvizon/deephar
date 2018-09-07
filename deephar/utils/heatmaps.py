import numpy as np

from .math import normalpdf2d
from .pose import get_visible_joints

class HeatMaps2D():
    def __init__(self, poses, numbins, variance=0.3):
        assert (poses.shape[-1] == 2) or ((poses.shape[-1] == 3)), \
                'Poses are expected to by 2D or 3D!'
        self.poses = poses
        if len(poses.shape) == 4:
            self.num_frames = poses.shape[1]
        else:
            self.num_frames = None

        self.numbins = numbins
        self.variance = variance
        self.num_joints = int(poses.shape[-2])

    def __getitem__(self, key):
        p = self.poses[key]
        if isinstance(key, int):
            return pose_heatmaps(p, self.numbins, self.num_joints,
                    variance=self.variance, num_frames=self.num_frames)
        if isinstance(key, slice):
            indices = key.indices(len(self))
            key = range(*indices)
        x = np.zeros((len(key),) + self.shape[1:])
        for i in range(len(key)):
            x[i,:] = pose_heatmaps(p[i], self.numbins, self.num_joints,
                    variance=self.variance, num_frames=self.num_frames)
        return x


    def __len__(self):
        return len(self.poses)

    @property
    def shape(self):
        if self.num_frames is None:
            return (len(self),) + (self.numbins, self.numbins, self.num_joints)
        else:
            return (len(self),) + (self.num_frames,
                    self.numbins, self.numbins, self.num_joints)


def pose_heatmaps(p, num_bins, num_joints, variance=0.1, num_frames=None):
    if num_frames is None:
        h = np.zeros((num_bins, num_bins, num_joints))
        v = get_visible_joints(p[:, 0:2])
        points = num_bins * p[:, 0:2]
        for j in range(num_joints):
            if v[j]:
                h[:,:,j] = normalpdf2d(num_bins,
                        points[j,0], points[j,1], variance)
    else:
        h = np.zeros((num_frames, num_bins, num_bins, num_joints))
        for f in range(num_frames):
            v = get_visible_joints(p[f][:, 0:2])
            points = num_bins * p[f][:, 0:2]
            for j in range(num_joints):
                if v[j]:
                    h[f,:,:,j] = normalpdf2d(num_bins,
                            points[j,0], points[j,1], variance)
    return h

