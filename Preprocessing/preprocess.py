import rasterio
import numpy as np

# Adding in the preprocessing functions
def minmax_scale(arr, min_val=-1., max_val=1.):
    minimum = np.nanmin(arr)
    maximum = np.nanmax(arr)
    arr = ((arr - minimum)/(maximum - minimum))*(max_val - min_val) + min_val
    return arr
    
def crop(band_sar, crop_size = 400):

    new_width = int((int(band_sar.shape[0] // crop_size) + 1) * crop_size)
    new_height = int((int(band_sar.shape[1] // crop_size) + 1) * crop_size)
    new_band_sar = np.zeros((new_width, new_height)).astype('float32')
    new_band_sar[:band_sar.shape[0], :band_sar.shape[1]] = band_sar

    num_sampled_0 = int(new_band_sar.shape[0] // crop_size)
    num_sampled_1 = int(new_band_sar.shape[1] // crop_size)
    itera_0 = 0
    itera_1 = 0
    patches_sar = []
    for idx0 in range(num_sampled_0):
        itera_1 = 0
        for idx1 in range(num_sampled_1):
          new_band_sar_1 = new_band_sar[itera_0 : itera_0 + crop_size, itera_1 :itera_1 + crop_size]
          patches_sar.append(new_band_sar_1)
          itera_1 = itera_1 + crop_size
        itera_0 = itera_0 + crop_size
    return patches_sar, new_width, new_height    

def stitch_patches(patches, width, height):
    patch_size = patches[0].shape[0]
    num_sampled_0 = int(width // patch_size)
    num_sampled_1 = int(height // patch_size)
    stitched_img = np.zeros((width, height)).astype('float32')

    patch_cnt = 0
    itera_0 = 0
    itera_1 = 0
    for idx0 in range(num_sampled_0):
        itera_1 = 0
        for idx1 in range(num_sampled_1):
          stitched_img[itera_0 : itera_0 + patch_size, itera_1 :itera_1 + patch_size] = patches[patch_cnt]
          itera_1 = itera_1 + patch_size
          patch_cnt += 1
        itera_0 = itera_0 + patch_size
    return stitched_img