import copy

import gzip
import numpy as np

from deephar.utils.io import printcn
from deephar.utils.io import HEADER
from deephar.utils.io import OKBLUE

# from deephar.utils.camera import Camera

TEST_MODE = 0
TRAIN_MODE = 1
VALID_MODE = 2


class BaseParser(object):

    compute_dataset_info = True
    avg_num_frames = 0
    pose_min = np.array([np.inf, np.inf, np.inf])
    pose_max = np.array([-np.inf, -np.inf, -np.inf])

    def __init__(self, fid):
        self.fid = fid

    def getattr(self):
        line = self.read_valid_line()
        if line is None:
            raise EOFError('File consumed!')

        return self.get_par_attr(line)

    def get_par_attr(self, line):
        val = line.split(':')
        assert len(val) == 2, 'Wrong "parameter: attributes," content'

        attr = val[1].split(',')
        if ((attr[-1] == '') or (attr[-1] == '\n')):
            del attr[-1]

        for i in range(len(attr)):
            attr[i] = attr[i].strip()

        return val[0], attr


    def read_valid_line(self):
        while True:
            line = self.fid.readline()
            if line:
                s = line.decode('utf-8')
                if ((s[0] != '\0') and (s[0] != '\n') and (s[0] != '#')):
                    return s
            else:
                return None

class BaseElement(object):
    def __init__(self, parent=None):
        self.parent = parent

    def copy(self):
        return copy.deepcopy(self)

    def get_camera(self):
        if hasattr(self, 'camera'):
            return self.camera
        elif hasattr(self, 'parent'):
            if self.parent is not None:
                return self.parent.get_camera()
        return None


class Entity(BaseElement):
    def __init__(self, parser, dim, num_joints, parent):
        BaseElement.__init__(self, parent=parent)
        self.pos = np.nan * np.ones((num_joints, dim))
        self.vis = np.nan * np.ones((num_joints, 1))
        self.mode = -1

        while True:
            par, attr = parser.getattr()

            if par == 'action_id':
                self.action_id = int(attr[0])

            if par == 'viewpoint_id':
                self.viewpoint_id = int(attr[0])

            if par == 'scale':
                self.scale = float(attr[0])

            if par == 'objpos':
                assert 2 == len(attr)
                self.objpos = np.array([float(attr[0]), float(attr[1])])

            if par == 'head':
                assert 4 == len(attr)
                self.head = np.array([float(attr[0]), float(attr[1]),
                                      float(attr[2]), float(attr[3])])

            if par == 'x':
                assert num_joints == len(attr)
                for i in range(num_joints):
                    self.pos[i, 0] = float(attr[i])

            if par == 'y':
                assert num_joints == len(attr)
                for i in range(num_joints):
                    self.pos[i, 1] = float(attr[i])

            if par == 'z':
                assert num_joints == len(attr)
                for i in range(num_joints):
                    self.pos[i, 2] = float(attr[i])

            if par == 'v':
                assert num_joints == len(attr)
                for i in range(num_joints):
                    self.vis[i, 0] = float(attr[i])

            if par == 'mode':
                self.mode = int(attr[0])
                break

        if BaseParser.compute_dataset_info:
            pmin = np.nan * np.ones((3,))
            pmax = np.nan * np.ones((3,))
            pmin[0:dim] = np.nanmin(self.pos, axis=0)
            pmax[0:dim] = np.nanmax(self.pos, axis=0)
            BaseParser.pose_min = \
                    np.nanmin(np.array([pmin, BaseParser.pose_min]), axis=0)
            BaseParser.pose_max = \
                    np.nanmax(np.array([pmax, BaseParser.pose_max]), axis=0)


class ImageFrame(BaseElement):
    def __init__(self, parser, dim, num_joints, parent=None):
        BaseElement.__init__(self, parent=parent)
        self.mode = -1

        while True:
            par, attr = parser.getattr()

            if par == 'image':
                self.image = attr[0]

            if par == 'res':
                self.res = np.array([float(attr[0]), float(attr[1])])

            if par == 'num_ent':
                self.num_ent = int(attr[0])
                self.entities = []
                for i in range(self.num_ent):
                    self.entities.append(
                            Entity(parser, dim, num_joints, parent=self))

            if par == 'mode':
                self.mode = int(attr[0])
                break


class SequenceOld(BaseElement):
    def __init__(self, parser, dim, num_joints):
        BaseElement.__init__(self)
        self.mode = -1

        while True:
            par, attr = parser.getattr()

            # if par == 'camera_parameters':
                # self.camera = Camera(attr)

            if par == 'num_frames':
                self.num_frames = int(attr[0])
                if BaseParser.compute_dataset_info:
                    BaseParser.avg_num_frames += self.num_frames
                self.frames = []
                for i in range(self.num_frames):
                    self.frames.append(ImageFrame(parser, dim, num_joints,
                        parent=self))

            if par == 'mode':
                self.mode = int(attr[0])
                break


def std_dat_parser(anno_obj, fid):
    parser = BaseParser(fid)

    while True:
        try:
            par, attr = parser.getattr()
        except Exception as e:
            print ('std_dat_parser: ' + str(e))
            break

        if par == 'action_labels':
            anno_obj.action_labels = attr

        if par == 'joint_labels':
            anno_obj.joint_labels = attr

        if par == 'viewpoint_labels':
            anno_obj.viewpoint_labels = attr

        if par == 'num_joints':
            anno_obj.num_joints = int(attr[0])

        if par == 'dim':
            anno_obj.dim = int(attr[0])

        if par == 'num_sequences':
            anno_obj.num_sequences = int(attr[0])
            anno_obj.sequences = []
            for i in range(anno_obj.num_sequences):
                anno_obj.sequences.append(
                        SequenceOld(parser, anno_obj.dim, anno_obj.num_joints))
            BaseParser.avg_num_frames /= len(anno_obj.sequences)

class Annotation(object):
    def __init__(self, dataset_path=None, custom_parser=None):
        self.sequences = []
        if custom_parser is None:
            assert dataset_path, \
                    "If a custom parser is not given, dataset_path is required"
        try:
            if custom_parser is not None:
                self.action_labels, \
                        self.joint_labels, \
                        self.viewpoint_labels,\
                        self.sequences = custom_parser(dataset_path)
            else:
                # Standard parser
                filename = '%s/annotations.dat.gz' % dataset_path
                fid = gzip.open(filename, 'r')
                gz_header = fid.readline()
                std_dat_parser(self, fid)
                fid.close()

            if BaseParser.compute_dataset_info:
                printcn(HEADER, '## Info on dataset "%s" ##' % dataset_path)
                printcn(OKBLUE, '  Average number of frames: %.0f' % \
                        BaseParser.avg_num_frames)
                printcn(OKBLUE, '  Min pose values on X-Y-Z: {}'.format(
                    BaseParser.pose_min))
                printcn(OKBLUE, '  Max pose values on X-Y-Z: {}'.format(
                    BaseParser.pose_max))

        except Exception as e:
            print ('Catch exception in Annotation class: ' + str(e))


def appstr(s, a):
    """Safe appending strings."""
    try:
        return s + a
    except:
        return None

