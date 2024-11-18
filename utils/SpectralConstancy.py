import numpy as np
from scipy import ndimage
from torch.nn import functional as F
import torch
import scipy

def to_float32(I):
  MAX = np.max(I)
  I = I/MAX
  return I.astype('float32')

def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn) / mx) * 255
    return I.astype(np.uint8)

def extract_unshuffled_tokens(image, patch_size):
    K,rows, cols = image.shape
    tokens = []
    for r in range(0, rows, patch_size):
        for c in range(0, cols, patch_size):
            tokens.append(image[:,r:r+patch_size, c:c+patch_size])
    return tokens

def tokens_to_image(tokens, image_shape):
    channels,rows, cols = image_shape
    patch_size = tokens[0].shape[1]
    reshaped = np.zeros(image_shape, dtype=np.float32)
    idx = 0
    for r in range(0, rows-patch_size, patch_size):
        for c in range(0, cols-patch_size, patch_size):
            reshaped[:,r:r+patch_size, c:c+patch_size] = tokens[idx]
            idx += 1
    return reshaped

def find_nearest_multiple(size, patch_size):
    return (size // patch_size) * patch_size

def raw_patches2cube(patches,coordinates,patch_size=5, image_size=(1020,1020)):
    raw_image = np.zeros((image_size[0], image_size[1]))
    counter = 0
    for cnt in range(len(coordinates)):
                raw_image[coordinates[cnt][0]:(coordinates[cnt][0] + patch_size),coordinates[cnt][1]:(coordinates[cnt][1] + patch_size)] = patches[counter]
                counter += 1
    return raw_image


def max_spectral(raw_image,pattern_size):
    shuffle = torch.nn.PixelShuffle(int(pattern_size))
    unshuffled_raw = F.pixel_unshuffle(torch.from_numpy(np.array(raw_image)).unsqueeze(0), pattern_size).detach().numpy()
    corrected_raw = np.copy(np.array(unshuffled_raw))
    unshuffled_raw  = np.asarray([scipy.ndimage.median_filter(unshuffled_raw[i,:,:], size=5) for i in range(unshuffled_raw.shape[0])])
    max_values = torch.from_numpy(np.max(unshuffled_raw,axis=(1,2))).detach().numpy()
    #
    corrected_raw = np.asarray([corrected_raw[i,:,:] / max_values[i] for i in range(unshuffled_raw.shape[0])])
    corrected_raw = shuffle(torch.from_numpy(corrected_raw)).squeeze(0).detach().numpy()
    return corrected_raw,max_values.ravel()

def max_spectral_fully_defined(cube):
    m,n,K = cube.shape
    filtered_cube  = np.asarray([scipy.ndimage.median_filter(cube[:,:,i], size=5) for i in range(K)])
    max_values  = np.max(filtered_cube,axis=(1,2))
    corrected_cube = np.asarray([filtered_cube[i,:,:] / max_values[i] for i in range(K)])
    corrected_cube = corrected_cube.transpose(1, 2, 0)
    #
    return corrected_cube,max_values.ravel()
















