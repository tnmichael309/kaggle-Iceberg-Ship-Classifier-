import numpy as np
from skimage.transform import warp, AffineTransform
from PIL import Image

class RandomAffineTransform(object):
    def __init__(self,
                 scale_range = None,
                 rotation_range = None,
                 shear_range = None,
                 translation_range = None
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    # img can be an PIL image with values in [0,1]
    def __call__(self, img):
        img_data = np.array(img)
            
        h, w, n_chan = img_data.shape
        
        scale = None
        rotation = None
        shear = None
        translation = None
        
        if self.scale_range is not None:
            #scale_x = np.random.uniform(*self.scale_range)
            #scale_y = np.random.uniform(*self.scale_range)
            scaling = np.random.uniform(*self.scale_range)
            scale = (scaling, scaling)
        
        if self.rotation_range is not None:
            rotation = np.random.uniform(*self.rotation_range)
            #rotation = self.rotation_range[np.random.randint(low=0, high=len(self.rotation_range))]
            
        if self.shear_range is not None:
            shear = np.random.uniform(*self.shear_range)
            
        if self.translation_range is not None:
            translation = (
                np.random.uniform(*self.translation_range) * w,
                np.random.uniform(*self.translation_range) * h
            )
            
        af = AffineTransform(scale=scale, shear=shear, rotation=rotation, translation=translation)
        new_img_data = warp(img_data, af.inverse)
        new_img = Image.fromarray(np.uint8(new_img_data * 255))
        
        return new_img