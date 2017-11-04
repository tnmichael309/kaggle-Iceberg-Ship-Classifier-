'''
This is a denoise script
'''

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, img_as_ubyte, color
from skimage.util import random_noise

def get_denoised_img(img):
    img = np.array(img)
    sigma_est = estimate_sigma(img, multichannel=True, average_sigmas=True)
    denoised_img = denoise_bilateral(img, sigma_color=sigma_est*.9, sigma_spatial=10, multichannel=True)
    denoised_img *= 255.0
    denoised_img = denoised_img.astype(np.uint8)
    
    return denoised_img
    

def denoise(df, store_path):
    
    denoised_imgs = []

    with ProcessPoolExecutor(max_workers=6) as executor:
        for i, denoised_img in enumerate(executor.map(get_denoised_img, df['band_mixed'])):
            print("\r{:.2f}% finished                 ".format(i/df.shape[0]*100.), end="")
            denoised_imgs.append(denoised_img)
        
        '''
        futures = [executor.submit(get_denoised_img, img) for img in df['band_mixed']]
        for i, f in enumerate(as_completed(futures)):
            print("\r{:.2f}% finished                 ".format(i/df.shape[0]*100.), end="")
            denoised_imgs.append(f.result())
        '''
        
    df['band_mixed'] = [img for img in denoised_imgs]
    df.to_json(store_path)
    
if __name__ == '__main__':
    train = pd.read_json('Data/processed_train.json')
    denoise(train, 'Data/denoised_processed_train.json')
    del train
    
    test = pd.read_json('Data/processed_test.json')
    denoise(test, 'Data/denoised_processed_test.json')
    del test