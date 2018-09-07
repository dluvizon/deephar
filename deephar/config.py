import numpy as np

# Input frame configuration

class DataConfig(object):
    def __init__(self, crop_resolution=(256, 256), image_num_channels=(3,),
            angles=[0], fixed_angle=0,
            scales=[1], fixed_scale=1,
            trans_x=[0], fixed_trans_x=0,
            trans_y=[0], fixed_trans_y=0,
            hflips=[0, 1], fixed_hflip=0,
            chpower=0.01*np.array(range(90, 110+1, 5)), fixed_chpower=1,
            subsampling=[1], fixed_subsampling=1):

        self.crop_resolution = crop_resolution
        self.image_num_channels = image_num_channels
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
        self.subsampling = subsampling
        self.fixed_subsampling = fixed_subsampling

    def get_fixed_config(self):
        return (self.fixed_angle,
                self.fixed_scale,
                self.fixed_trans_x,
                self.fixed_trans_y,
                self.fixed_hflip,
                self.fixed_chpower,
                self.fixed_subsampling)

    def random_data_generator(self):
        angle = DataConfig._getrand(self.angles)
        scale = DataConfig._getrand(self.scales)
        trans_x = DataConfig._getrand(self.trans_x)
        trans_y = DataConfig._getrand(self.trans_y)
        hflip = DataConfig._getrand(self.hflips)
        chpower = [1, 1, 1]
        chpower[0] = DataConfig._getrand(self.chpower)
        chpower[1] = DataConfig._getrand(self.chpower)
        chpower[2] = DataConfig._getrand(self.chpower)
        subsampling = DataConfig._getrand(self.subsampling)

        return angle, scale, trans_x, trans_y, hflip, chpower, subsampling

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
        trans_x=np.array(range(-30, 31, 5)),
        trans_y=np.array(range(-5, 6, 1)),
        subsampling=[1, 2]
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

