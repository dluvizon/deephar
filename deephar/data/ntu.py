import os
import copy

import numpy as np
from PIL import Image

from deephar.data.datasets import get_clip_frame_index
from deephar.utils import *

use_small_images = True
image_prefix = 'images-small' if use_small_images else 'images'
video_subsample = 2

ACTION_LABELS = ['drink water', 'eat meal/snack', 'brushing teeth',
        'brushing hair', 'drop', 'pickup', 'throw', 'sitting down',
        'standing up (from sitting position)', 'clapping', 'reading',
        'writing', 'tear up paper', 'wear jacket', 'take off jacket',
        'wear a shoe', 'take off a shoe', 'wear on glasses',
        'take off glasses', 'put on a hat/cap', 'take off a hat/cap',
        'cheer up', 'hand waving', 'kicking something',
        'put something inside pocket / take out something from pocket',
        'hopping (one foot jumping)', 'jump up',
        'make a phone call/answer phone', 'playing with phone/tablet',
        'typing on a keyboard', 'pointing to something with finger',
        'taking a selfie', 'check time (from watch)', 'rub two hands together',
        'nod head/bow', 'shake head', 'wipe face', 'salute',
        'put the palms together', 'cross hands in front (say stop)',
        'sneeze/cough', 'staggering', 'falling', 'touch head (headache)',
        'touch chest (stomachache/heart pain)', 'touch back (backache)',
        'touch neck (neckache)', 'nausea or vomiting condition',
        'use a fan (with hand or paper)/feeling warm',
        'punching/slapping other person', 'kicking other person',
        'pushing other person', 'pat on back of other person',
        'point finger at the other person', 'hugging other person',
        'giving something to other person', 'touch other person s pocket',
        'handshaking', 'walking towards each other',
        'walking apart from each other']

JOINT_LABELS = ['base of the spine', 'middle of the spine', 'neck', 'head',
        'left shoulder', 'left elbow', 'left wrist', 'left hand',
        'right shoulder', 'right elbow', 'right wrist', 'right hand',
        'left hip', 'left knee', 'left ankle', 'left foot', 'right hip',
        'right knee', 'right ankle', 'right foot', 'spine',
        'tip of the left hand', 'left thumb', 'tip of the right hand',
        'right thumb']

VIEWPOINT_LABELS = ['cam1', 'cam2', 'cam3']


def serialize_index_sequences(sequences):
    frame_idx = []
    for s in range(len(sequences)):
        for f in range(len(sequences[s])):
            frame_idx.append((s, f))

    return frame_idx


def ntu_load_annotations(dataset_path, eval_mode='cs',
        num_S=17, num_C=3, num_P=40, num_R=2, num_A=60):

    # Saniry checks
    assert eval_mode in ['cs', 'cv'], \
        'Invalid evaluation mode {}'.format(eval_mode)
    
    ntud_numpy_dir = os.path.join(dataset_path, 'nturgb+d_numpy')
    ntud_images_dir = os.path.join(dataset_path, image_prefix)
    for d in [ntud_numpy_dir, ntud_images_dir]:
        assert os.path.isdir(d), \
            f'Error: check your NTU dataset! `{d}` not found!'

    min_num_frames = np.inf
    max_num_frames = -np.inf
    num_videos = [0, 0, 0]

    cs_train = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,
            25, 27, 28, 31, 34, 35, 38]
    cv_train = [2, 3]

    sequences = [[], [], []]
    seq_ids = [[], [], []]
    actions = [[], [], []]

    for s in range(1,num_S+1):
        for c in range(1,num_C+1):
            for p in range(1,num_P+1):
                for r in range(1,num_R+1):
                    for a in range(1,num_A+1):

                        sequence_id = \
                                'S%03dC%03dP%03dR%03dA%03d' % (s, c, p, r, a)
                        filename = os.path.join(ntud_numpy_dir,
                                sequence_id + '.npy')
                        if not os.path.isfile(filename):
                            continue # Ignore missing annotation files

                        if eval_mode == 'cs':
                            mode = TRAIN_MODE if p in cs_train else TEST_MODE
                        else:
                            mode = TRAIN_MODE if c in cv_train else TEST_MODE

                        data = np.load(filename)
                        if video_subsample is not None:
                            data = data[0::video_subsample, :]

                        """Compute some stats about the dataset."""
                        if len(data) < min_num_frames:
                            min_num_frames = len(data)
                        if len(data) > max_num_frames:
                            max_num_frames = len(data)
                        num_videos[mode] += 1

                        sequences[mode].append(data)
                        seq_ids[mode].append(sequence_id)
                        actions[mode].append(a)

    frame_idx = [serialize_index_sequences(sequences[0]),
            serialize_index_sequences(sequences[1]), []]

    printcn('', 'Max/Min number of frames: {}/{}'.format(
        max_num_frames, min_num_frames))
    printcn('', 'Number of videos: {}'.format(num_videos))

    return sequences, frame_idx, seq_ids, actions


class Ntu(object):
    def __init__(self, dataset_path, dataconf, poselayout=pa20j3d,
            topology='sequence', use_gt_bbox=False, remove_outer_joints=True,
            clip_size=16, pose_only=False, num_S=17):

        self.dataset_path = dataset_path
        self.dataconf = dataconf
        self.poselayout = poselayout
        self.topology = topology
        self.use_gt_bbox = use_gt_bbox
        self.clip_size = clip_size
        self.remove_outer_joints = remove_outer_joints
        self.pose_only = pose_only
        self.action_labels = ACTION_LABELS
        self.joint_labels = JOINT_LABELS

        try:
            self.sequences, self.frame_idx, self.seq_ids, self.actions = \
                    ntu_load_annotations(dataset_path, num_S=num_S)
        except:
            warning('Error loading NTU RGB+D dataset!')
            raise

    def get_data(self, key, mode, frame_list=None, bbox=None):
        """Method to load NTU samples specified by mode and key,
        do data augmentation and bounding box cropping.
        """
        output = {}

        if mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
            random_clip = True
        else:
            dconf = self.dataconf.get_fixed_config()
            random_clip = False

        if self.topology == 'sequences':
            seq_idx = key
            seq = self.sequences[mode][seq_idx]
            seq_id = self.seq_ids[mode][seq_idx]
            act = self.actions[mode][seq_idx]
            if frame_list == None:
                frame_list = get_clip_frame_index(len(seq), dconf['subspl'],
                        self.clip_size, random_clip=random_clip)
        else:
            seq_idx, frame_idx = self.frame_idx[mode][key]
            seq = self.sequences[mode][seq_idx]
            seq_id = self.seq_ids[mode][seq_idx]
            act = self.actions[mode][seq_idx]
            frame_list = [frame_idx]

        objframes = seq[frame_list]

        """Load pose annotation"""
        pose, visible = self.get_pose_annot(objframes)

        if use_small_images:
            w, h = (int(1920/2), int(1080/2))
        else:
            w, h = (1920, 1080)

        """Compute the ground truth bounding box, if not given"""
        if bbox is None:
            if self.use_gt_bbox:
                bbox = get_gt_bbox(pose[:, :, 0:2], visible, (w, h),
                        scale=dconf['scale'], logkey=key)
            else:
                bbox = objposwin_to_bbox(np.array([w / 2, h / 2]),
                        (dconf['scale']*max(w, h), dconf['scale']*max(w, h)))

        rootz = np.nanmean(pose[:, 0, 2])
        if np.isnan(rootz):
            rootz = np.nanmean(pose[:, :, 2], axis=(0, 1))

        zrange = np.array([rootz - dconf['scale']*1000,
            rootz + dconf['scale']*1000])

        objpos, winsize = bbox_to_objposwin(bbox)
        if min(winsize) < 32:
            winsize = (32, 32)
        objpos += dconf['scale'] * np.array([dconf['transx'], dconf['transy']])

        """Pre-process data for each frame"""
        if self.pose_only:
            frames = None
        else:
            frames = np.zeros((len(objframes),) + self.dataconf.input_shape)

        for i in range(len(objframes)):
            if self.pose_only:
                imgt = T(None, img_size=(w, h))
            else:
                imagepath = os.path.join(self.dataset_path, image_prefix,
                        seq_id, '%05d.jpg' % objframes[i][0])
                imgt = T(Image.open(imagepath))

            imgt.rotate_crop(dconf['angle'], objpos, winsize)
            imgt.resize(self.dataconf.crop_resolution)

            if dconf['hflip'] == 1:
                imgt.horizontal_flip()

            imgt.normalize_affinemap()
            if not self.pose_only:
                frames[i, :, :, :] = normalize_channels(imgt.asarray(),
                        channel_power=dconf['chpower'])

            pose[i, :, 0:2] = transform_2d_points(imgt.afmat, pose[i, :, 0:2],
                    transpose=True)
            pose[i, :, 2] = (pose[i, :, 2] -zrange[0]) / (zrange[1] -zrange[0])

            if imgt.hflip:
                pose[i, :, :] = pose[i, self.poselayout.map_hflip, :]

        """Set outsider body joints to invalid (-1e9)."""
        pose = np.reshape(pose, (-1, self.poselayout.dim))
        pose[np.isnan(pose)] = -1e9
        v = np.expand_dims(get_visible_joints(pose[:,0:2]), axis=-1)
        pose[(v==0)[:,0],:] = -1e9
        pose = np.reshape(pose, (len(objframes), self.poselayout.num_joints,
            self.poselayout.dim))
        v = np.reshape(v, (len(objframes), self.poselayout.num_joints, 1))

        pose = np.concatenate((pose, v), axis=-1)
        if self.topology != 'sequences':
            pose = np.squeeze(pose, axis=0)
            if not self.pose_only:
                frames = np.squeeze(frames, axis=0)

        action = np.zeros(self.get_shape('ntuaction'))
        action[act - 1] = 1.

        output['seq_idx'] = seq_idx
        output['frame_list'] = frame_list
        output['ntuaction'] = action
        output['pennaction'] = np.zeros((15,))
        output['pose'] = pose
        output['frame'] = frames

        """Take the last transformation matrix, it should not change"""
        output['afmat'] = imgt.afmat.copy()

        return output


    def get_pose_annot(self, frames):

        num_joints = len(JOINT_LABELS)
        pose = frames[:, 1+3*num_joints:]

        p = np.zeros((len(frames), num_joints, self.poselayout.dim))

        if use_small_images:
            p[:, :, 0] = pose[:, 0:num_joints] / 2.
            p[:, :, 1] = pose[:, num_joints:2*num_joints] / 2.
        else:
            p[:, :, 0] = pose[:, 0:num_joints]
            p[:, :, 1] = pose[:, num_joints:2*num_joints]

        if self.poselayout.dim == 3:
            p[:, :, 2] = pose[:, 2*num_joints:]

        p = p[:, self.poselayout.map_from_ntu, :].copy()
        v = np.apply_along_axis(lambda x: 1 if x.all() else 0,
                axis=2, arr=(p > 0))
        p[v==0, :] = np.nan

        return p, v

    def get_clip_index(self, key, mode, subsamples=[2]):
        assert self.topology == 'sequences', 'Topology not supported'

        seq = self.sequences[mode][key]
        index_list = []
        for sub in subsamples:
            start_frame = 0
            while True:
                last_frame = start_frame + self.clip_size * sub
                if last_frame > len(seq):
                    break
                index_list.append(range(start_frame, last_frame, sub))
                start_frame += int(self.clip_size / 2) + (sub - 1)

        return index_list


    def clip_length(self):
        if self.topology == 'sequences':
            return self.clip_size
        else:
            return None

    def clip_shape(self):
        if self.topology == 'sequences':
            return (self.clip_size,)
        else:
            return ()

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return self.clip_shape() + self.dataconf.input_shape
        if dictkey == 'pose':
            return self.clip_shape() \
                    + (self.poselayout.num_joints, self.poselayout.dim+1)
        if dictkey == 'ntuaction':
            return (len(self.action_labels),)
        if dictkey == 'pennaction':
            return (15,)
        if dictkey == 'afmat':
            return (3, 3)
        raise Exception('Invalid dictkey on get_shape!')

    def get_length(self, mode):
        if self.topology == 'sequences':
            return len(self.sequences[mode])
        else:
            return len(self.frame_idx[mode])

