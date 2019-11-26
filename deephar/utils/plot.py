import numpy as np
from PIL import Image

from deephar.utils.io import WARNING
from deephar.utils.io import FAIL
from deephar.utils.io import printcn

from deephar.utils.pose import pa16j2d
from deephar.utils.pose import pa17j3d
from deephar.utils.pose import pa20j3d
from deephar.utils.colors import hex_colors

try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
except Exception as e:
    printcn(FAIL, str(e))
    plt = None


def data_to_image(x, gray_scale=False):
    """ Convert 'x' to a RGB Image object.

    # Arguments
        x: image in the format (num_cols, num_rows, 3) for RGB images or
            (num_cols, num_rows) for gray scale images. If None, return a
            light gray image with size 100x100.
        gray_scale: convert the RGB color space to a RGB gray scale space.
    """

    if x is None:
        x = 224 * np.ones((100, 100, 3), dtype=np.uint8)

    if x.max() - x.min() > 0.:
        buf = 255. * (x - x.min()) / (x.max() - x.min())
    else:
        buf = x.copy()

    if len(buf.shape) == 3:
        (w, h) = buf.shape[0:2]
        num_ch = buf.shape[2]
    else:
        (h, w) = buf.shape
        num_ch = 1

    if ((num_ch is 3) and gray_scale):
        g = 0.2989*buf[:,:,0] + 0.5870*buf[:,:,1] + 0.1140*buf[:,:,2]
        buf[:,:,0] = g
        buf[:,:,1] = g
        buf[:,:,2] = g
    elif num_ch is 1:
        aux = np.zeros((h, w, 3), dtype=buf.dtype)
        aux[:,:,0] = buf
        aux[:,:,1] = buf
        aux[:,:,2] = buf
        buf = aux

    return Image.fromarray(buf.astype(np.uint8), 'RGB')


def show(x, gray_scale=False, jet_cmap=False, filename=None):
    """ Show 'x' as an image on the screen.
    """
    if jet_cmap is False:
        img = data_to_image(x, gray_scale=gray_scale)
    else:
        if plt is None:
            printcn(WARNING, 'pyplot not defined!')
            return
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=x.min(), vmax=x.max())
        img = cmap(norm(x))
    if filename:
        plt.imsave(filename, img)
    else:
        plt.imshow(img)
        plt.show()


def draw(x=None,
        skels=[],
        bboxes=[],
        bbox_color='g',
        abs_pos=False,
        plot3d=False,
        single_window=False,
        figsize=(16,9),
        axis='on',
        facecolor='white',
        azimuth=65,
        dpi=100,
        filename=None):

    # Configure the ploting environment
    if plt is None:
        printcn(WARNING, 'pyplot not defined!')
        return

    """ Plot 'x' and draw over it the skeletons and the bounding boxes.
    """
    img = data_to_image(x)
    if abs_pos:
        w = None
        h = None
    else:
        w,h = img.size

    def add_subimage(f, subplot, img):
        ax = f.add_subplot(subplot)
        plt.imshow(img, zorder=-1)
        return ax

    fig = [plt.figure(figsize=figsize)]
    ax = []

    if plot3d:
        if single_window:
            ax.append(add_subimage(fig[0], 121, img))
            ax.append(fig[0].add_subplot(122, projection='3d'))
        else:
            ax.append(add_subimage(fig[0], 111, img))
            fig.append(plt.figure(figsize=figsize))
            ax.append(fig[1].add_subplot(111, projection='3d'))
    else:
        ax.append(add_subimage(fig[0], 111, img))

    plt.axis(axis)

    # Plotting skeletons if not None
    if skels is not None:
        if isinstance(skels, list) or len(skels.shape) == 3:
            for s in skels:
                plot_skeleton_2d(ax[0], s, h=h, w=w)
            if plot3d:
                plot_3d_pose(s, subplot=ax[-1], azimuth=azimuth)
        else:
            plot_skeleton_2d(ax[0], skels, h=h, w=w)
            if plot3d:
                plot_3d_pose(skels, subplot=ax[-1], azimuth=azimuth)

    # Plotting bounding boxes if not None
    if bboxes is not None:
        if isinstance(bboxes, list) or len(bboxes.shape) == 3:
            for b, c in zip(bboxes, bbox_color):
                _plot_bbox(ax[0], b, h=h, w=w, c=c, lw=4)
        else:
            _plot_bbox(ax[0], bboxes, h=h, w=w, c=bbox_color, lw=4)


    if filename:
        fig[0].savefig(filename, bbox_inches='tight', pad_inches=0,
                facecolor=facecolor, dpi=dpi)
        if plot3d and (single_window is False):
            fig[-1].savefig(filename + '.eps',
                    bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    for i in range(len(fig)):
        plt.close(fig[i])


def _get_poselayout(num_joints):
    if num_joints == 16:
        return pa16j2d.color, pa16j2d.cmap, pa16j2d.links
    elif num_joints == 17:
        return pa17j3d.color, pa17j3d.cmap, pa17j3d.links
    elif num_joints == 20:
        return pa20j3d.color, pa20j3d.cmap, pa20j3d.links


def plot_3d_pose(pose, subplot=None, filename=None, color=None, lw=3,
        azimuth=65):

    if plt is None:
        raise Exception('"matplotlib" is required for 3D pose plotting!')

    num_joints, dim = pose.shape
    assert dim in [2, 3], 'Invalid pose dimension (%d)' % dim
    assert ((num_joints == 16) or (num_joints == 17)) or (num_joints == 20), \
            'Unsupported number of joints (%d)' % num_joints

    col, cmap, links = _get_poselayout(num_joints)
    if color is None:
        color = col

    def _func_and(x):
        if x.all():
            return 1
        return 0

    points = np.zeros((num_joints, 3))
    for d in range(dim):
        points[:,d] = pose[:,d]
    for i in range(num_joints):
        points[i, 2] = max(0, points[i, 2])

    valid = np.apply_along_axis(_func_and, axis=1, arr=(points[:,0:2] > -1e6))

    if subplot is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None
        ax = subplot

    for j in range(num_joints):
        if valid[j]:
            x, y, z = points[j]
            ax.scatter([z], [x], [y], lw=lw, c=color[cmap[j]])

    for i in links:
        if valid[i[0]] and valid[i[1]]:
            c = color[cmap[i[0]]]
            ax.plot(points[i, 2], points[i, 0], points[i, 1], c=c, lw=lw)

    ax.view_init(10, azimuth)
    ax.set_aspect('equal')
    ax.set_xlabel('Z (depth)')
    ax.set_ylabel('X (width)')
    ax.set_zlabel('Y (height)')
    ax.set_xlim([0, 1.])
    ax.set_ylim([0, 1.])
    ax.set_zlim([0, 1.])
    plt.gca().invert_xaxis()
    plt.gca().invert_zaxis()

    if fig is not None:
        if filename:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close(fig)


def _plot_bbox(subplot, bbox, h=None, w=None, scale=16, lw=2, c=None):
    assert len(bbox) == 4

    b = bbox.copy()
    if w is not None:
       b[0] *= w
       b[2] *= w
    if h is not None:
       b[1] *= h
       b[3] *= h

    if c is None:
        c = hex_colors[np.random.randint(len(hex_colors))]

    x = np.array([b[0], b[2], b[2], b[0], b[0]])
    y = np.array([b[1], b[1], b[3], b[3], b[1]])
    subplot.plot(x, y, lw=lw, c=c, zorder=1)


def plot_skeleton_2d(subplot, skel, h=None, w=None,
        joints=True, links=True, scale=16, lw=4):

    s = skel.copy()
    num_joints = len(s)
    assert ((num_joints == 16) or (num_joints == 17)) or (num_joints == 20), \
            'Unsupported number of joints (%d)' % num_joints

    color, cmap, links = _get_poselayout(num_joints)

    x = s[:,0]
    y = s[:,1]
    v = s > -1e6
    v = v.any(axis=1).astype(np.float32)

    # Convert normalized skeletons to image coordinates.
    if w is not None:
        x *= w
    if h is not None:
        y *= h

    if joints:
        for i in range(len(v)):
            if v[i] > 0:
                c = color[cmap[i]]
                subplot.scatter(x=x[i], y=y[i], c=c, lw=lw, s=scale, zorder=2)

    if links:
        for i in links:
            if ((v[i[0]] > 0) and (v[i[1]] > 0)):
                c = color[cmap[i[0]]]
                subplot.plot(x[i], y[i], lw=lw, c=c, zorder=1)


# def drawhm(hm, zero_clip=False, vmax=None, filename=None):
    # #heatmaps = np.transpose(heatmaps, (0, 3, 1, 2))
    # fb = hm.copy()

    # if zero_clip:
        # fb = (fb > 0) * fb

    # vmin = fb.min()
    # if vmax is None:
        # vmax = fb.max()

    # print (vmin, vmax)

    # cmap = plt.cm.jet
    # norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # image = cmap(norm(fb))
    # print (filename)
    # if filename is not None:
        # plt.imsave(filename, image)
    # else:
        # plt.show(image)
    # plt.close()

