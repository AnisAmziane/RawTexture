import numpy as np
from numba import jit
def get_closest_wavelengths(sampling_w, scan):
    # get the closest wavelengths in scan array to match those in sampling_w array
    dist = np.abs(sampling_w[:, np.newaxis] - scan)
    potentialClosest = dist.argmin(axis=1)
    closest_bands, counts = np.unique(potentialClosest, return_counts=True)
    return closest_bands
def find_closest_spectra_illuminant(camera_wvs,illuminant_wvs):
    """ This function finds the shared wavelengths among camera SSFs and illuminant
       It returns the closet wavelength index in camera SSFs and the closest CIE color matching function values.
       Illuminant is Extended D65 by default.
       Inputs:
       camera_wvs: Wavelengths sampled by camera filters. Interpolation is not considered here.
       Outputs:
       Closest wavelength index: array
       """
    # ------ For camera side
    closestFound_1 = get_closest_wavelengths(illuminant_wvs, camera_wvs)
    # ------ For illulinant side
    closestFound_2 = get_closest_wavelengths(camera_wvs, illuminant_wvs)
    min_count = np.min([len(closestFound_1), len(closestFound_2)])
    closests_idx_camera_wvs = closestFound_1[:min_count]
    closests_idx_illu_wvs = closestFound_2[:min_count]
    return closests_idx_camera_wvs, closests_idx_illu_wvs

@jit(nopython=True)
def reflectance2radiance(reflectance_cube,illuminant):
    #---------------------------------------------------------------------------------------------------------------
    rows,cols,_ = reflectance_cube.shape
    radiance_cube = np.zeros((reflectance_cube.shape[0],reflectance_cube.shape[1],reflectance_cube.shape[2]), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            radiance_cube[i,j] = reflectance_cube[i, j] * illuminant
    return radiance_cube



