import cv2
import numpy as np
import math
import sys
sys.path.append('/home/anis/PycharmProjects/ms_project/venv/Codes/')
from skimage import feature
from numba import jit

@jit(nopython=True)
def get_pixel(img, center, x, y):
    #value = 0
    if img[x][y] >= center:
        value = 1
    else:
        value = 0
    return value

@jit(nopython=True)
def lbp_calculated_pixel(img, x, y):
    ''' clockwise neighbor weights for LBP computation at pixel P
     64 | 128 |   1
    ----------------
     32 |  P  |   2
    ----------------
     16 |   8 |   4
    '''
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
    val_ar.append(get_pixel(img, center, x, y + 1))  # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y))  # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
    val_ar.append(get_pixel(img, center, x, y - 1))  # left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
    val_ar.append(get_pixel(img, center, x - 1, y))  # top
    weights = [1, 2, 4, 8, 16, 32, 64, 128]
    code = 0 # LBP code in a given square 3x3 neighborhood
    for i in range(len(val_ar)):
        code += val_ar[i] * weights[i]
    return code

@jit(nopython=True)
def simpleLBP(img,distance):
    rows,cols = img.shape
    lbp_codes = []
    for i in range(0+distance,rows-1):
        for j in range(0+distance,cols-1):
            code = lbp_calculated_pixel(img, i, j)
            lbp_codes.append(code)
    return np.asarray(lbp_codes)


@jit(nopython=True)
def MLBP_codes(img,pattern_image,dist):
    rows,cols = img.shape
    bands = np.unique(pattern_image)
    B_codes = []
    for b in bands:
        this_band_codes = []
        for i in range(0 + dist, rows-dist, dist):
            for j in range(0 + dist, cols-dist, dist):
                if pattern_image[i,j] == b:
                    code = lbp_calculated_pixel(img, i, j)
                    this_band_codes.append(code)
        B_codes.append(np.asarray(this_band_codes))
    return B_codes

def concatenate_hists(B_codes):
    hists = []
    for codes in B_codes:
        # h = np.zeros(256,dtype=np.uint8)
        (h, _) = np.histogram(np.asarray(codes), bins=256, range=(0, 255))
        # h = h / h.sum(dtype=np.float32)
        hists.append(h)
    concatenated_hists = np.hstack(hists)
    return concatenated_hists


@jit(nopython=True)
def bilinear_interpolate_numpy(im, x, y):

    """ Bilinear interpolate pixel valu of 2d image at new x,y coordinate
        and returns the interpolated pixel value

    """
    lowerBound = 0
    X_upperBound = im.shape[0]-1
    Y_upperBound = im.shape[1]-1
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    x0 = max(lowerBound, min(x0, X_upperBound))
    y0 = max(lowerBound, min(y0, Y_upperBound))
    x1 = max(lowerBound, min(x1, X_upperBound))
    y1 = max(lowerBound, min(y1, Y_upperBound))
    #
    Ia = im[ x0, y0 ]
    Ib = im[ x0, y1 ]
    Ic = im[ x1, y0 ]
    Id = im[ x1, y1 ]
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    interpolated_value = Ia*wa + Ib*wb + Ic*wc + Id*wd
    return interpolated_value

@jit(nopython=True)
def bilinear_interpolate_numpy3D(im, x, y):
    """ Bilinear interpolate pixel values of 3d image at new x,y coordinate
        and returns the interpolated pixel vector
    """
    K  = im.shape[2]
    interpolated_pixel_vector = np.zeros(K,dtype=np.float64)
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    for k in range(K):
        Ia = im[x0, y0,k]
        Ib = im[x0, y1,k]
        Ic = im[x1, y0,k]
        Id = im[x1, y1,k]
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)
        interpolated_value = Ia * wa + Ib * wb + Ic * wc + Id * wd
        interpolated_pixel_vector[k] = interpolated_value
    return np.asarray(interpolated_pixel_vector)


@jit(nopython=True)
def get_neighbors(patch,i, j,radius):
    neighbors_values = []
    neighbors_position = [[i-radius, j+radius],[i, j+radius],[i+radius, j+radius],[i+radius, j],[i+radius, j-radius], [i, j-radius],[i-radius, j-radius], [i-radius, j]]
    for a in neighbors_position:
        neighbors_values.append(patch[a[0],a[1]])
    return neighbors_values

@jit(nopython=True)
def get_neighbors_spectra(patch,i, j,distance):
    neighbors_values = []
    neighbors_position = [[i-distance, j+distance],[i, j+distance],[i+distance, j+distance],[i+distance, j],[i+distance, j-distance], [i, j-distance],[i-distance, j-distance], [i-distance, j]]
    for a in neighbors_position:
        neighbor = patch[a[0], a[1], :]
        neighbor = np.asarray(neighbor).astype(np.float64)
        neighbors_values.append(neighbor)
    return neighbors_values



@jit(nopython=True)
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

@jit(nopython=True)
def thresholded_angles(center, reference, neighbours):
    out = []
    Nbpixels = len(neighbours)
    for a in range(Nbpixels):
        if angle_between(neighbours[a],reference) >= angle_between(center,reference):
            out.append(1)
        else:
            out.append(0)
    return out

@jit(nopython=True)
def thresholded_norms(center, neighbours):
    out = []
    Nbpixels = len(neighbours)
    norm_center = euclidean_norm(center)
    for a in range(Nbpixels):
        norm_neighbour = euclidean_norm(neighbours[a])
        if norm_neighbour>=norm_center:
            out.append(1)
        else:
            out.append(0)
    return out

@jit(nopython=True)
def inner_product(v1, v2):  # inner_product(x, x)
    inner_product = 0
    for i in range(len(v1)):
        inner_product += v1[i] * v2[i]
    return inner_product


@jit(nopython=True)
def euclidean_norm(v1):  # euclidean_norm(x)
    norm = 0
    for i in range(len(v1)):
        norm += v1[i] * v1[i]
    return np.sqrt(norm)

@jit(nopython=True)
def angle_between(v1, v2):  # angle = angle_between(x, y)
    # epsilon = 0.0000001
    v1 = np.asarray(v1,dtype=np.float64)
    v2 = np.asarray(v2,dtype=np.float64)
    product = inner_product(v1,v2)
    normV1 = euclidean_norm(v1)
    normV2 = euclidean_norm(v2)
    norm = (normV1*normV2)
    if norm ==0:
        angle=0
    else:
        angle = np.arccos(product/norm)
    return angle


@jit(nopython=True)
def _bit_rotate_right(value, length):
    """Cyclic bit shift to the right.
    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer
    """
    return (value >> 1) | ((value & 1) << (length - 1))

@jit(nopython=True)
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out


# from mahotas.features.lbp import *
def get_concatenated_hists(patch_3D,P,radius,lbp_type="default"):
   ### concatenate lbp histograms of each channel
   if lbp_type == 'default':
       bins = 2 ** P
       upperbound = (2 ** P) - 1
   if lbp_type == 'uniform':
       bins = P * (P - 1) + 3
       upperbound = P * (P - 1) + 2
   m,n,K= patch_3D.shape # get size of patches
   current_patch = patch_3D
   concatenated_hists = []
   for k in range(K):
        lbp_codes = feature.local_binary_pattern(current_patch[:,:,k], P, radius, method=lbp_type)
        (hist, counts) = np.histogram(lbp_codes.ravel(), bins=bins, range=(0, upperbound))
        hist_normalized = hist / hist.sum(dtype=np.float32)
        concatenated_hists.append(hist_normalized)
   concatenated_hists = np.hstack(concatenated_hists)
   return concatenated_hists


def get_concatenated_hists_simple(image, P,distance):
   ### concatenate lbp histogram of each channel
   bins = 2 ** P
   upperbound = (2 ** P) - 1
   m,n,K= image.shape # get size of patches
   concatenated_hists = []
   for k in range(K):
        lbp_codes = simpleLBP(image[:,:,k],distance)
        (hist, counts) = np.histogram(lbp_codes.ravel(), bins=bins, range=(0, upperbound))
        # hist_normalized = hist / hist.sum(dtype=np.float32)
        concatenated_hists.append(hist)
   concatenated_hists = np.hstack(concatenated_hists)
   return concatenated_hists

@jit(nopython=True)
def marginal_lbp_codes(patch, P, radius,lbp_type='default'):  # get_marginal_lbp_codes(patch, 8, 1,lbp_type='default')
    lines = patch.shape[0]
    columns = patch.shape[1]
    K = patch.shape[2]
    list_codes = []
    start = 0 + radius
    end_lines = lines - radius
    end_columns = columns - radius
    for k in range(K):
        k_lbp_codes = []
        for i in range(start, end_lines, 1):
            for j in range(start, end_columns, 1):
                # if patch[i, j,0] != 0: # to avoid background pixels and edge pixels
                #     if (patch[(i - radius), (j - radius), 0] != 0) and (
                #             patch[(i - radius), (j + radius), 0] != 0) and (
                #             patch[(i + radius), (j - radius), 0] != 0) and (
                #             patch[(i + radius), (j + radius), 0] != 0):
                            neighbors_values = np.zeros((P), dtype=np.float64)
                            for neighbor in range(P):
                                angle = ((2 * math.pi) * neighbor) / P
                                i_neighbors = (i + radius * math.cos(angle))
                                j_neighbors = (j - radius * math.sin(angle))
                                if (angle % (math.pi / 2) == 0):
                                    neighbors_values[neighbor] = patch[int(round(i_neighbors)), int(round(j_neighbors)),k]
                                else:
                                    neighbors_values[neighbor] = bilinear_interpolate_numpy(patch[:,:,k], i_neighbors, j_neighbors)
                                central_value = patch[i, j,k]
                                codes = thresholded(central_value, neighbors_values)
                                if lbp_type == 'default':
                                    value = 0
                                    for p in range(P):
                                        value += codes[p] * (2 ** p)
                                    k_lbp_codes.append(value)
                                if lbp_type == 'uniform':
                                    changes = 0
                                    for p in range(P - 1):
                                        changes += (codes[p] - codes[p + 1]) != 0
                                    value = 0
                                    if changes <= 2:
                                        for p in range(P):
                                            value += codes[p]
                                    else:
                                        value = P * (P - 1) + 2
                                    k_lbp_codes.append(value)
        list_codes.append(np.asarray(k_lbp_codes))
    return list_codes


@jit(nopython=True)
def simple_lbp(patch, P, radius,lbp_type='default'):  # get_marginal_lbp_codes(patch, 8, 1,lbp_type='default')
    lines = patch.shape[0]
    columns = patch.shape[1]
    # list_codes = []
    start = radius
    end_lines = lines - radius
    end_columns = columns - radius
    lbp_codes = []
    for i in range(start, end_lines, radius):
        for j in range(start, end_columns, radius):
            # if patch[i, j] != 0: # to avoid background pixels and edge pixels
            #      if (patch[(i - radius), (j - radius)] != 0) and (
            #             patch[(i - radius), (j + radius)] != 0) and (
            #             patch[(i + radius), (j - radius)] != 0) and (
            #             patch[(i + radius), (j + radius)] != 0):
                        neighbors_values = get_neighbors(patch, i, j, radius)
                        central_value = patch[i, j]
                        codes = thresholded(central_value, neighbors_values)
                        if lbp_type == 'default':
                            value = 0
                            for p in range(P):
                                value += codes[p] * (2 ** p)
                            lbp_codes.append(value)
                        if lbp_type == 'uniform':
                            changes = 0
                            for p in range(P - 1):
                                changes += (codes[p] - codes[p + 1]) != 0
                            value = 0
                            if changes <= 2:
                                for p in range(P):
                                    value += codes[p]
                            else:
                                value = P * (P - 1) + 2
                            lbp_codes.append(value)
    return lbp_codes

@jit(nopython=True)
def ppi_img(cube):
    lines = cube.shape[0]
    columns = cube.shape[1]
    ppi = np.zeros((lines,columns),dtype=np.float64)
    for i in range(lines):
        for j in range(columns):
            ppi[i,j] = np.mean(cube[i,j,:])
    return ppi

def simple_lbp_ppi_hist(cube,distance):
    ppi = ppi_img(cube)
    lbp_codes = simpleLBP(ppi,distance)
    hist, _= np.histogram(np.asarray(lbp_codes), bins=256, range=(0, 255))
    return hist


@jit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin

@jit(nopython=True)
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges

def np_histogram(a,bins,normalize=True):
    hist, counts = np.histogram(a, bins=256, range=(0, bins-1))
    if normalize:
        hist = hist/np.sum(hist)
    return hist




@jit(nopython=True)
def local_angles_pattern(patch, P, radius,reference='avg',interpolate=True,lbp_type='default'):
    patch = patch.astype(np.float64)
    lines = patch.shape[0]
    columns = patch.shape[1]
    lbp_codes = []
    K = patch.shape[2]
    start = 0 + radius
    end_lines = lines - radius
    end_columns = columns - radius
    for i in range(start, end_lines):
        for j in range(start, end_columns):
                    reference_spectrum = np.zeros((K), dtype=np.float64)
                    neighbors_values = np.zeros((P, K), dtype=np.float64)
                    if interpolate:
                        for neighbor in range(P):
                                angle = ((2 * math.pi) * neighbor) / P
                                i_neighbors = (i + radius * math.cos(angle))
                                j_neighbors = (j - radius * math.sin(angle))
                                if (angle % (math.pi / 2) == 0):
                                    neighbors_values[neighbor, :] = patch[int(round(i_neighbors)), int(round(j_neighbors)), :]
                                else:
                                    neighbors_values[neighbor, :] = bilinear_interpolate_numpy3D(patch, i_neighbors, j_neighbors)
                    else:
                        neighbors_list = get_neighbors_spectra(patch, i, j, radius)
                        for neighbor in range(P):
                            neighbors_values[neighbor, :] = neighbors_list[neighbor]
                    if reference == 'avg':
                        for k in range(K):
                            reference_spectrum[k] = np.mean(neighbors_values[:, k])
                    if reference == 'median':
                        for k in range(K):
                            reference_spectrum[k] = np.median(neighbors_values[:, k])
                    central_spectrum = patch[i, j, :]
                    codes = thresholded_angles(central_spectrum, reference_spectrum,neighbors_values)
                    if lbp_type == 'default':
                        value = 0
                        for p in range(P):
                            value += codes[p] * (2 ** p)
                        lbp_codes.append(value)
                    if lbp_type == 'uniform':
                        changes = 0
                        for p in range(P - 1):
                            changes += (codes[p] - codes[p + 1]) != 0
                        value = 0
                        if changes <= 2:
                            for p in range(P):
                                value += codes[p]
                        else:
                            value = P*(P-1)+ 2
                        lbp_codes.append(value)
    return lbp_codes


@jit(nopython=True)
def lcc(patch, P, distance,interpolate=True,reference='avg'):

    patch = patch.astype(np.float64)
    lines = patch.shape[0]
    columns = patch.shape[1]
    K = patch.shape[2]
    lcc_patterns = []# = np.zeros((lines, columns), dtype=np.uint8)
    start = 0 + distance
    end_lines = lines - distance
    end_columns = columns - distance
    for i in range(start, end_lines):
        for j in range(start, end_columns):
                reference_spectrum = np.zeros((K), dtype=np.float64)
                neighbors_values = np.zeros((P, K), dtype=np.float64)
                if interpolate:
                    for neighbor in range(P):
                        angle = ((2 * math.pi) * neighbor) / P
                        i_neighbors = (i + distance * math.cos(angle))
                        j_neighbors = (j - distance * math.sin(angle))
                        if (angle % (math.pi / 2) == 0):
                            neighbors_values[neighbor, :] = patch[int(round(i_neighbors)), int(round(j_neighbors)), :]
                        else:
                            neighbors_values[neighbor, :] = bilinear_interpolate_numpy3D(patch, i_neighbors, j_neighbors)
                else:
                    neighbors_list = get_neighbors_spectra(patch, i, j, distance)
                    for neighbor in range(P):
                        neighbors_values[neighbor, :] = neighbors_list[neighbor]
                if reference == 'avg':
                    for k in range(K):
                        reference_spectrum[k] = np.mean(neighbors_values[:, k])
                if reference == 'median':
                    for k in range(K):
                        reference_spectrum[k] = np.median(neighbors_values[:, k])
                central_spectrum = patch[i, j, :]
                arccos = angle_between(central_spectrum, reference_spectrum)
                if arccos> (math.pi/4):
                     arccos = math.pi/4
                arccos = int(np.round(arccos*255))
                lcc_patterns.append(arccos)
    return lcc_patterns

