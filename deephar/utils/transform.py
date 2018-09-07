import numpy as np
from PIL import Image


class AffineTransform(object):
    def __init__(self):
        self.afmat = np.eye(3)

    def _apply(self, t):
        self.afmat = np.dot(t, self.afmat)

    def scale(self, w, h):
        t = np.eye(3)
        t[0,0] *= w
        t[1,1] *= h
        self._apply(t)

    def translate(self, x, y):
        t = np.eye(3)
        t[0,2] = x
        t[1,2] = y
        self._apply(t)

    def rotate(self, angle, center):
        self.translate(-center[0], -center[1])
        self.rotate_center(angle)
        self.translate(center[0], center[1])

    def rotate_center(self, angle):
        t = np.eye(3)
        angle *= np.pi / 180
        a = np.cos(angle)
        b = np.sin(angle)
        t[0,0] = a
        t[0,1] = b
        t[1,1] = a
        t[1,0] = -b
        self._apply(t)

    def affine_hflip(self):
        t = np.eye(3)
        t[0,0] = -1
        self._apply(t)


class T(AffineTransform):
    def __init__(self, img, img_size=None):
        self.img = img
        if img_size is not None:
            self.img_size = tuple(img_size)
        else:
            self.img_size = img_size
        self.hflip = False
        AffineTransform.__init__(self)

    def resize(self, size, resample=Image.BILINEAR):
        t = self.scale(size[0] / self.size[0], size[1] / self.size[1])
        if self.img is not None:
            self.img = self.img.resize(size, resample)
        else:
            self.img_size = tuple(size)

    def normalize_affinemap(self):
        t = self.scale(1 / self.size[0], 1 / self.size[1])

    def crop(self, box):
        t = self.translate(-box[0], -box[1])
        if self.img is not None:
            self.img = self.img.crop(box)
        else:
            self.img_size = (box[2] - box[0], box[3] - box[1])

    def rotate_crop(self, angle, center, winsize,
            resample=Image.BILINEAR):
        """Rotate, crop, and resize the image.

        # Arguments
            angle: angle to rotate in degrees.
            winsize: (w, h) window size to crop in the input image.
            center: center point (x,y) to rotate from, None to use the
                image center.
            resample: rescaling method, according to PIL.Image.
        """

        if center is None:
            center = (self.size[0]/2, self.size[1]/2)

        if angle != 0:
            self.rotate(angle, center)

        # Compute the margins after rotation
        corners = np.array([
            [0, 0],
            [self.size[0], 0],
            [0, self.size[1]],
            [self.size[0], self.size[1]]
            ]).transpose()
        corners = transform_2d_points(self.afmat, corners)

        # Translate to zero margin
        self.translate(-min(corners[0,:]), -min(corners[1,:]))

        # Rotate image
        if (self.img is not None) and (angle != 0):
            self.img = self.img.rotate(angle, resample, expand=True)

        center = np.array([center[0], center[1]])
        center = transform_2d_points(self.afmat, center)

        crop = np.array([center[0] - winsize[0]/2, center[1] - winsize[1]/2,
            center[0] + winsize[0]/2, center[1] + winsize[1]/2], dtype=int)
        self.crop(crop)


    def horizontal_flip(self):
        self.affine_hflip()
        self.translate(self.size[0], 0)
        if self.img is not None:
            self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        self.hflip = not self.hflip

    def asarray(self, dtype=np.float32):
        if self.img is not None:
            return np.asarray(self.img, dtype=dtype)
        else:
            return np.zeros(self.img_size + (3,))

    @property
    def size(self):
        if self.img is not None:
            return self.img.size
        else:
            return self.img_size


def transform_2d_points(A, x, transpose=False, inverse=False):
    """Apply a given affine transformation to 2D points.

    # Arguments
        A: [3, 3] affine transformation map: T(x) = Ax.
        x: [dim, N] points (normal case, otherwise, set the flag 'transpose').
        transpose: flag to be setted if 'x' is [N, dim].
        inverse: flag to apply the inverse transformation on A.

    # Return
        The transformed points.
    """

    squeeze = False
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=-1)
        squeeze = True
    elif transpose:
        x = np.transpose(x)

    (dim, N) = x.shape
    assert (dim == 2), \
            'transform_2d_points: Only 2D points are supported, get ' +str(dim)

    if inverse:
        A = np.linalg.inv(A)

    y = np.ones((dim+1, N))
    y[0:dim,:] = x[0:dim,:]
    y = np.dot(A, y)[0:dim]

    if squeeze:
        return np.squeeze(y)
    if transpose:
        return np.transpose(y)
    return y


def transform_pose_sequence(A, poses, inverse=True):
    """For each pose in a sequence, apply the given affine transformation.

    # Arguments
        A: [3, 3] affine transformation matrix or
           [num_samples, 3, 3] matrices.
        poses: [num_samples, num_points, dim] vector of pose sequences.
        inverse: flag to apply the inverse transformation on A.

    # Return
        The transformed points.
    """

    assert (len(poses.shape) == 3), \
            'transform_pose_sequence: expected 3D tensor, got ' \
            + str(poses.shape)

    if len(A.shape) == 3:
        assert len(A) == len(poses), \
                'A is ' + str(A.shape) + ' and poses is ' + str(poses.shape)

    if inverse:
        if len(A.shape) == 3:
            for i in range(len(A)):
                A[i] = np.linalg.inv(A[i])
        else:
            A = np.linalg.inv(A)

    y = np.empty(poses.shape)
    for j in range(len(poses)):
        if len(A.shape) == 3:
            y[j, :, :] = transform_2d_points(A[j], poses[j], transpose=True)
        else:
            y[j, :, :] = transform_2d_points(A, poses[j], transpose=True)

    return y


def normalize_channels(frame, channel_power=1):

    if type(channel_power) is not int:
        assert len(channel_power) == 3, 'channel_power expected to be int or ' \
                + 'tuple/list with len=3, {} given.'.format(channel_power)

    frame /= 255.

    if type(channel_power) is int:
        if channel_power != 1:
            frame = np.power(frame, channel_power)
    else:
        for c in range(3):
            if channel_power[c] != 1:
                frame[:,:, c] = np.power(frame[:,:, c], channel_power[c])

    frame -= .5
    frame *= 2.

    return frame

