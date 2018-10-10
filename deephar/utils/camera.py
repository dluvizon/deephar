import numpy as np

from deephar.utils.transform import transform_pose_sequence


class Camera(object):
    """Camera implementation.

    # Arguments
        R: Rotation matrix (3,3)
        t: Translation vector world coordinate system (3, 1)
        f: Focal length (1, 2)
        c: Principal point (1, 2)
        p: Skew (1, 2)
        k: Distortion coefficients (3,), frequently not required.

    # TODO
        Implement distortion coefficients.
    """

    def __init__(self, R, t, f, c, p, k=None):
        self.R = R
        self.R_inv = np.linalg.inv(self.R)
        self.t = np.reshape(t, (3, 1))
        self.f = np.reshape(f, (1, 2))
        self.c = np.reshape(c, (1, 2))
        self.p = np.reshape(p, (1, 2))
        self.k = k
        if self.k is not None:
            self.k = np.reshape(self.k, (3,))

    def project(self, points_w):
        """Projects world 3D points (in millimeters) to the image coordinate
        system (in x-y pixels and depth).
        """
        assert len(points_w.shape) == 2 and points_w.shape[1] == 3, \
                'Invalid shape for points_w ({}),'.format(points_w.shape) \
                + 'expected (n, 3)'

        x = np.matmul(self.R, points_w.T - self.t).T
        x[:, 0:2] /= x[:, 2:3]

        if self.k is not None:
            r2, radial, tan = get_r2_radial_tan(x[:, 0:2], self.k, self.p)
            x[:, 0:2] *= np.expand_dims(radial + tan, axis=-1)
            x[:, 0:2] += np.dot(np.expand_dims(r2, axis=-1), self.p)

        x[:, 0:2] = x[:, 0:2]*self.f + self.c

        return x

    def inverse_project(self, points_uvd):
        """Projects a point in the camera coordinate system (x-y in pixels and
        depth) to world 3D coordinates (in millimeters).
        """
        assert len(points_uvd.shape) == 2 and points_uvd.shape[1] == 3, \
                'Invalid shape for points_uvd ({}),'.format(points_uvd.shape) \
                + ' expected (n, 3)'

        x = points_uvd.copy()
        x[:, 0:2] = (x[:, 0:2] - self.c) / self.f

        if self.k is not None:
            r2, radial, tan = get_r2_radial_tan(x[:, 0:2], self.k, self.p)
            x[:, 0:2] -= np.dot(np.expand_dims(r2, axis=-1), self.p)
            x[:, 0:2] /= np.expand_dims(radial + tan, axis=-1)

        x[:, 0:2] *= x[:, 2:3]
        x = (np.matmul(self.R_inv, x.T) + self.t).T

        return x

    def serialize(self):
        s = np.array(self.R).reshape((9,))
        s = np.concatenate([s, np.array(self.t).reshape((3,))])
        s = np.concatenate([s, np.array(self.f).reshape((2,))])
        s = np.concatenate([s, np.array(self.c).reshape((2,))])
        s = np.concatenate([s, np.array(self.p).reshape((2,))])
        if self.k is not None:
            s = np.concatenate([s, self.k])

        return s

def get_r2_radial_tan(x, k, p):
    """Given a set o points x [num_points, 2] in the image coordinate system,
    compute the required vectors to apply the distortion coefficients.
    """
    assert x.ndim == 2 and x.shape[1] == 2
    assert k.shape == (3,) and p.shape == (1, 2)

    r2 = np.power(x[:, 0], 2) + np.power(x[:, 1], 2)
    radial = 1. + r2*k[0] + np.power(r2, 2)*k[1] + np.power(r2, 3)*k[2]
    tan = np.sum(x * p, axis=-1)

    return r2, radial, tan


def camera_deserialize(s):
    R, s = np.split(s, [9])
    t, s = np.split(s, [3])
    f, s = np.split(s, [2])
    c, s = np.split(s, [2])
    p, s = np.split(s, [2])

    k = None
    if len(s) > 0:
        k, s = np.split(s, [3])

    return Camera(np.reshape(R, (3, 3)), t, f, c, p, k)


def project_pred_to_camera(pred, afmat, resol_z, root_z):
    num_samples, num_joints, dim = pred.shape
    root_z = np.expand_dims(root_z, axis=-1)

    proj = np.zeros(pred.shape)
    proj[:,:,0:2] = transform_pose_sequence(afmat, pred[:,:,0:2], inverse=True)
    proj[:,:,2] = (resol_z * (pred[:,:,2] - 0.5)) + root_z

    return proj

