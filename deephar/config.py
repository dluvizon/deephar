import numpy as np

import keras.backend as K
K.set_image_data_format('channels_last')

class DataConfig(object):
    """Input frame configuration and data augmentation setup."""

    def __init__(self, crop_resolution=(256, 256), image_channels=(3,),
            angles=[0], fixed_angle=0,
            scales=[1], fixed_scale=1,
            trans_x=[0], fixed_trans_x=0,
            trans_y=[0], fixed_trans_y=0,
            hflips=[0, 1], fixed_hflip=0,
            chpower=0.01*np.array(range(90, 110+1, 2)), fixed_chpower=1,
            geoocclusion=None, fixed_geoocclusion=None,
            subsampling=[1], fixed_subsampling=1):

        self.crop_resolution = crop_resolution
        self.image_channels = image_channels
        if K.image_data_format() == 'channels_last':
            self.input_shape = crop_resolution + image_channels
        else:
            self.input_shape = image_channels + crop_resolution
        self.angles = angles
        self.fixed_angle = fixed_angle
        self.scales = scales
        self.fixed_scale = fixed_scale
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.fixed_trans_x = fixed_trans_x
        self.fixed_trans_y = fixed_trans_y
        self.hflips = hflips
        self.fixed_hflip = fixed_hflip
        self.chpower = chpower
        self.fixed_chpower = fixed_chpower
        self.geoocclusion = geoocclusion
        self.fixed_geoocclusion = fixed_geoocclusion
        self.subsampling = subsampling
        self.fixed_subsampling = fixed_subsampling

    def get_fixed_config(self):
        return {'angle': self.fixed_angle,
                'scale': self.fixed_scale,
                'transx': self.fixed_trans_x,
                'transy': self.fixed_trans_y,
                'hflip': self.fixed_hflip,
                'chpower': self.fixed_chpower,
                'geoocclusion': self.fixed_geoocclusion,
                'subspl': self.fixed_subsampling}

    def random_data_generator(self):
        angle = DataConfig._getrand(self.angles)
        scale = DataConfig._getrand(self.scales)
        trans_x = DataConfig._getrand(self.trans_x)
        trans_y = DataConfig._getrand(self.trans_y)
        hflip = DataConfig._getrand(self.hflips)
        chpower = (DataConfig._getrand(self.chpower),
                DataConfig._getrand(self.chpower),
                DataConfig._getrand(self.chpower))
        geoocclusion = self.__get_random_geoocclusion()
        subsampling = DataConfig._getrand(self.subsampling)

        return {'angle': angle,
                'scale': scale,
                'transx': trans_x,
                'transy': trans_y,
                'hflip': hflip,
                'chpower': chpower,
                'geoocclusion': geoocclusion,
                'subspl': subsampling}

    def __get_random_geoocclusion(self):
        if self.geoocclusion is not None:

            w = int(DataConfig._getrand(self.geoocclusion) / 2)
            h = int(DataConfig._getrand(self.geoocclusion) / 2)
            xmin = w + 1
            xmax = self.crop_resolution[0] - xmin
            ymin = h + 1
            ymax = self.crop_resolution[1] - ymin

            x = DataConfig._getrand(range(xmin, xmax, 5))
            y = DataConfig._getrand(range(ymin, ymax, 5))
            bbox = (x-w, y-h, x+w, y+h)

            return bbox

        else:
            return None

    @staticmethod
    def _getrand(x):
        return x[np.random.randint(0, len(x))]


# Data generation and configuration setup

mpii_sp_dataconf = DataConfig(
        crop_resolution=(256, 256),
        angles=np.array(range(-40, 40+1, 5)),
        scales=np.array([0.7, 1., 1.3]),
        )

pennaction_dataconf = DataConfig(
        crop_resolution=(256, 256),
        angles=np.array(range(-30, 30+1, 5)),
        scales=np.array([0.7, 1.0, 1.3, 2.5]),
        trans_x=np.array(range(-40, 40+1, 5)),
        trans_y=np.array(range(-40, 40+1, 5)),
        subsampling=[1, 2]
        )

pennaction_ar_dataconf = DataConfig(
        crop_resolution=(256, 256),
        angles=np.array(range(-30, 30+1, 5)),
        scales=np.array([0.7, 1.0, 1.3]),
        trans_x=np.array(range(-40, 40+1, 5)),
        trans_y=np.array(range(-40, 40+1, 5)),
        subsampling=[1, 2, 3],
        fixed_subsampling=2
        )

human36m_dataconf = DataConfig(
        crop_resolution=(256, 256),
        angles=np.array(range(-10, 10+1, 5)),
        scales=np.array([0.8, 1.0, 1.2]),
        geoocclusion=np.array(range(20, 90)),
        )

ntu_ar_dataconf = DataConfig(
        crop_resolution=(256, 256),
        angles=[0],
        scales=np.array([0.7, 1.0, 1.3]),
        trans_x=range(-40, 40+1, 5),
        trans_y=range(-40, 40+1, 5),
        subsampling=[1, 2, 3],
        fixed_subsampling=2
        )

ntu_pe_dataconf = DataConfig(
        crop_resolution=(256, 256),
        angles=np.array(range(-15, 15+1, 5)),
        scales=np.array([0.8, 1.0, 1.2, 2.0]),
        trans_x=np.array(range(-40, 40+1, 5)),
        trans_y=np.array(range(-10, 10+1, 5)),
        subsampling=[1, 2, 4]
        )

# Aliases.
mpii_dataconf = mpii_sp_dataconf
