# -*- coding: utf-8 -*-
""" Title: MSFA patterns and raw image simulation
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2022
  """
import numpy as np
import random
import numba
from numba import jit
import torch
from torch.nn import functional as F
import cv2
from albumentations import RandomRotate90, GridDistortion,RandomResizedCrop,OpticalDistortion,ElasticTransform, HorizontalFlip, VerticalFlip
from scipy import ndimage
import torchvision.transforms as transforms
np.random.seed(42)
random.seed(42)
#

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

def gaussian_noise(x, mean=0, var_limit=(0, 30), per_channel=False):
    import cv2
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    var = random.uniform(var_limit[0], var_limit[1])
    sigma = var ** 0.5
    random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
    if per_channel:
        gauss = random_state.normal(mean, sigma, x.shape)
    else:
        gauss = random_state.normal(mean, sigma, x.shape[:2])
        if len(x.shape) == 3:
            gauss = np.expand_dims(gauss, -1)
    noisy_x = x.astype("float32") + gauss
    # noisy_x = cv2.normalize(noisy_x, 0, 255, cv2.NORM_MINMAX, dtype=-1) #
    noisy_x = (noisy_x * 225).astype("uint8")
    return noisy_x


def one_hot_encode(label, label_values):
    label = torch.as_tensor(label).long()
    onehotmap = F.one_hot(label, num_classes=label_values)
    return np.array(onehotmap)
@jit(nopython=True)
def reorganize_cube(raw,pattern):
    rows,cols = raw.shape
    w,h = pattern.shape
    reshaped = np.zeros((rows-(h-1),cols-(w-1)),dtype=np.float32)
    for i in range(0,rows-(h-1)):
        for j in range(0,cols-(w-1)):
            window = raw[i:(i+h),j:(j+w)]
            tmp = []
            for ii in range(h):
                for jj in range(w):
                    tmp.append(window[ii,jj,pattern[ii,jj]])
            tmp = np.asarray(tmp)
            reshaped[i,j,:] = tmp
    return reshaped

def get_closest_band(sampling_w, scan):
    # get the closest wavelengths in scan array to match those in sampling_w array
    dist = np.abs(sampling_w[:, np.newaxis] - scan)
    closest = np.argmin(dist)
    closest_band = sampling_w[closest]
    return closest_band

# @jit(nopython=True)
def make_pattern(pattern_type='Imec16'):
    """ For CA80 5x5/4x4, and IMEC 5x5/4x4/2x2  MSFAs """
    assert pattern_type in ['Imec25','Imec16','Imec4','CA80_25','CA80_16','Simple25','Simple16','Bayer','X-Trans','Silios'], \
        f"Pattern not recognized, choose one of the following patterns:\n Imec25, Imec16, Imec4, CA80_25, CA80_16 , " \
                                                                                          f"Simple25 ,or Simple16'"
    if pattern_type =='Imec25':
        pattern = [[18, 19, 17, 16, 1],
                          [10, 11, 9, 8, 2],
                          [6, 7, 5, 4, 3],
                          [22, 23, 21, 20, 0],
                          [14, 15, 13, 12, 24]] ### Imec25 pattern
    if pattern_type == 'CA80_25':
        pattern = [[4, 24, 14, 23, 2],
                  [20, 19, 7, 3, 16],
                  [15, 22, 0, 21, 12],
                  [5, 17, 11, 1, 10],
                  [18, 6, 13, 8, 9]] # CA80 5x5 pattern inspired from Imec25 pattern. Because central bands associated to the channels are not
                  # in ascending order. For exemple channel 4 of a 25 channel CA80 cube corresponds to channel 18 if the channels where in ascending order.
                  # 25 channels in CA80 are provided by feature selection procedure. From most discriminant channel to least one
    if pattern_type =='Imec16':
        pattern = [[6, 7, 5, 4],
                   [14, 15, 13, 12],
                   [10, 11, 9, 8],
                   [2, 3, 1, 0]] ###
    if pattern_type == 'CA80_16':
        pattern = [[7, 8, 3, 15],
                    [5, 9, 11, 1],
                    [14, 4, 6, 13],
                    [12, 0, 2, 10]] # CA80 4x4 pattern inspired from Imec16 pattern. Same thing as for CA80_25

    if pattern_type =='Imec4':
        pattern = [[3,0],
                   [1,2]] # [[NIR, B],
                                #  [G, R]]

    if pattern_type =='Simple25':
        used_bands = np.arange(25)
        pattern = used_bands.reshape(5, 5)
    if pattern_type =='Simple16':
        used_bands = np.arange(16)
        pattern = used_bands.reshape(4, 4)
    if pattern_type =='Bayer':
        pattern = [[1, 0], [2, 1]]  # [[G, R],
        #  [B, G]]
    if pattern_type =='X-Trans': # 6x6 pattern #
        pattern = [[1, 2, 0, 1, 0, 2], #  [[G, B,R G,R,B],
                   [0, 1, 1, 2, 1, 1], #  [R,G,G,B,G,G],
                   [2, 1, 1, 0, 1, 1], #  [B,G,G,R,G,G],
                   [1, 0, 2, 1, 2, 0], #  [G,R,B,G,B,R],
                   [2, 1, 1, 0, 1, 1], #  [B,G,G,R,G,G],
                   [0, 1, 1, 2, 1, 1]] #  [R,G,G,B,G,G],]

    if pattern_type == 'Silios':  # 3x3 pattern #
        pattern = [[7, 6, 5],
                   [0, 8, 4],
                   [1, 2, 3]]

    pattern = np.asarray(pattern)

    return pattern

# @jit(nopython=True)
def get_mosaic(cube,dims,pattern):
    # pattern = make_pattern(pattern_type,bands=selected_bands)
    pattern_image = create_pattern_image(pattern, dims[0], dims[1])
    rows = cube.shape[0]
    cols = cube.shape[1]
    # mosaiced_img = np.zeros((rows, cols), dtype=np.float32)
    mosaiced_img = np.zeros((rows, cols))
    for l in range(rows):
        for c in range(cols):
            mosaiced_img[l,c] = cube[l,c,pattern_image[l,c]]
    return mosaiced_img

@jit(nopython=True)
def create_pattern_image(pattern, width, height):
    w, h = pattern.shape
    pattern_image = np.zeros((width, height), dtype=np.uint8)
    for l in range(0, width, w):
        for c in range(0, height, h):
            tmp = pattern_image[l:(l + w), c:(c + h)]
            if tmp.shape[:2] == (width, height):
                pattern_image[l:(l + w), c:(c + h)] = pattern
            else:
                xx, yy = pattern_image[l:(l + w), c:(c + h)].shape
                pattern_image[l:(l + w), c:(c + h)] = pattern[:xx, :yy]
    return pattern_image

class MakeDataset(torch.utils.data.Dataset):
  'Simple class to characterize a dataset for PyTorch'
  def __init__(self, list_patches,list_labels,patch_size):
        'Initialization'
        self.M, self.N, self.K = patch_size
        self.list_patches = list_patches
        self.list_labels = list_labels
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        patch = self.list_patches[index]
        label = self.list_labels[index]
        # # Debugging statements
        # print("Patch shape:", patch.shape)
        # print("Label:", label)
        patch = torch.from_numpy(np.asarray(patch.reshape((self.K, self.M, self.N)))).to(torch.float32)
        label = torch.from_numpy(np.asarray(label)).to(torch.long)
        return patch, label


## for augmentations
def extract_raw_patches(image, size):
  rows, cols = image.shape
  x_step = size[0]
  y_step = size[1]
  list_patches = []
  coordinates = []
  for i in range(0, rows, y_step):
      for j in range(0, cols, x_step):
          tmp = image[i:i + y_step, j:j + x_step]
          if tmp.shape[:2] == (x_step, y_step):
              list_patches.append(tmp)
              coordinates.append((i, j))
  return list_patches, coordinates

def extract_patches(image, size):
  rows, cols,K = image.shape
  x_step = size[0]
  y_step = size[1]
  list3D_patches = []
  coordinates = []
  for i in range(0, rows, y_step):
      for j in range(0, cols, x_step):
          tmp = image[i:i + y_step, j:j + x_step]
          if tmp.shape[:2] == (x_step, y_step):
              list3D_patches.append(tmp)
              coordinates.append((i, j))
  return list3D_patches, coordinates



def find_nearest_multiple(size, patch_size):
    return (size // patch_size) * patch_size

def extract_tokens(image, patch_size):
    max_pattern_pixels = find_nearest_multiple(image.shape[0], patch_size)
    image = image[:max_pattern_pixels,:max_pattern_pixels]
    rows, cols = image.shape
    tokens = []
    for r in range(0, rows, patch_size):
        for c in range(0, cols, patch_size):
            tokens.append(image[r:r+patch_size, c:c+patch_size])
    return tokens

def mix_tokens(tokens):
    np.random.seed(100)
    np.random.shuffle(tokens)
    # np.random.seed(55)
    # np.random.shuffle(tokens)
    np.random.seed(42)
    return tokens

def tokens_to_image(tokens, image_shape):
    rows, cols = image_shape
    patch_size = tokens[0].shape[0]
    mixed_image = np.zeros(image_shape, dtype=np.uint8)
    idx = 0
    for r in range(0, rows-patch_size, patch_size):
        for c in range(0, cols-patch_size, patch_size):
            mixed_image[r:r+patch_size, c:c+patch_size] = tokens[idx]
            idx += 1
    return mixed_image

def reshape_tokens(raw_image,tokens,coordinates,patch_size=5):
    raw_image = np.copy(raw_image)
    counter = 0
    for cnt in range(len(coordinates)):
                raw_image[coordinates[cnt][0]:(coordinates[cnt][0] + patch_size),coordinates[cnt][1]:(coordinates[cnt][1] + patch_size)] = tokens[counter]
                counter += 1
    return raw_image


def add_noise_(raw_image, noise_std):
    import numpy as np
    np.random.seed(42)
    #
    noise = np.random.normal(scale=noise_std, size=raw_image.shape)
    noisy_image = raw_image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def gaussian_noise(x, mean=0, var_limit=(0, 30), per_channel=False):
    import cv2
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    var = random.uniform(var_limit[0], var_limit[1])
    sigma = var ** 0.5
    random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
    if per_channel:
        gauss = random_state.normal(mean, sigma, x.shape)
    else:
        gauss = random_state.normal(mean, sigma, x.shape[:2])
        if len(x.shape) == 3:
            gauss = np.expand_dims(gauss, -1)
    noisy_x = x.astype("float32") + gauss
    noisy_x = cv2.normalize(noisy_x, 0, 255, cv2.NORM_MINMAX, dtype=-1) #
    noisy_x = (noisy_x * 225).astype("uint8")
    return noisy_x

def add_noise_cube(cube):
    raw_image = gaussian_noise(cube, mean=0, var_limit=(0, 30), per_channel=False)
    return raw_image

def apply_blur(raw_image,kernel_size):
    # import skimage as ski
    # blurred = ski.filters.gaussian(
    #     raw_image, sigma=(sigma, sigma), truncate=3.5)
    blurred_image = cv2.GaussianBlur(raw_image, (kernel_size, kernel_size), 0)
    return blurred_image

def grid_distorsion(raw_image,pattern_width):
    shuffle = torch.nn.PixelShuffle(int(pattern_width))
    unshuffled_raw = F.pixel_unshuffle(torch.from_numpy(raw_image).unsqueeze(0), pattern_width).detach().numpy()
    #
    aug = GridDistortion(p=1.0)
    augmented = aug(image=unshuffled_raw)
    x = augmented['image']
    x = shuffle(torch.from_numpy(x)).squeeze(0).detach().numpy()
    return x

def optical_distorsion(raw_image,pattern_width):
    shuffle = torch.nn.PixelShuffle(int(pattern_width))
    unshuffled_raw = F.pixel_unshuffle(torch.from_numpy(raw_image).unsqueeze(0), pattern_width).detach().numpy()
    #
    aug = OpticalDistortion(p=1.0)
    augmented = aug(image=unshuffled_raw)
    x = augmented['image']
    x = shuffle(torch.from_numpy(x)).squeeze(0).detach().numpy()
    return x


def elastic_transform(raw_image,pattern_width):
    shuffle = torch.nn.PixelShuffle(int(pattern_width))
    unshuffled_raw = F.pixel_unshuffle(torch.from_numpy(raw_image).unsqueeze(0), pattern_width).detach().numpy()
    #
    aug = ElasticTransform(p=1.0)
    augmented = aug(image=unshuffled_raw)
    x = augmented['image']
    x = shuffle(torch.from_numpy(x)).squeeze(0).detach().numpy()
    return x

def HFlip(raw_image,pattern_width):
    shuffle = torch.nn.PixelShuffle(int(pattern_width))
    unshuffled_raw = F.pixel_unshuffle(torch.from_numpy(raw_image).unsqueeze(0), pattern_width)
    x = HFlip_2(unshuffled_raw.permute(1, 2, 0).detach().numpy())
    x = shuffle(torch.from_numpy(x).permute(2, 0, 1)).squeeze(0).detach().numpy()
    return x

def VFlip(raw_image,pattern_width):
    shuffle = torch.nn.PixelShuffle(int(pattern_width))
    unshuffled_raw = F.pixel_unshuffle(torch.from_numpy(raw_image).unsqueeze(0), pattern_width)
    x = VFlip_2(unshuffled_raw.permute(1,2,0).detach().numpy())
    x = shuffle(torch.from_numpy(x).permute(2,0,1)).squeeze(0).detach().numpy()
    return x

def HFlip_2(cube):
    aug = HorizontalFlip(p=1.0)
    augmented = aug(image=cube)
    x = augmented['image']
    return x

def VFlip_2(cube):
    aug = VerticalFlip(p=1.0)
    augmented = aug(image=cube)
    x = augmented['image']
    return x


def translate(raw_image,pattern_width,rate=0.6,direction='vertical'):
    m,n = raw_image.shape
    if direction=='horizontal':
        nb_patterns = int(n/pattern_width)
        nb_displacement = int(np.floor(nb_patterns*rate))*pattern_width
        translated_image = torch.roll(torch.from_numpy(raw_image), shifts=(nb_displacement, 0), dims=(1, 0)).detach().numpy()

    if direction=='vertical':
        nb_patterns = int(m / pattern_width)
        nb_displacement = int(np.floor(nb_patterns*rate))*pattern_width
        translated_image = torch.roll(torch.from_numpy(raw_image), shifts=(0, nb_displacement),dims=(1, 0)).detach().numpy()
    if direction == 'vertical+horizontal':
        nb_patterns = int(m / pattern_width)
        nb_displacement = int(np.floor(nb_patterns*rate))*pattern_width
        translated_image = torch.roll(torch.from_numpy(raw_image), shifts=(nb_displacement, nb_displacement),
                                      dims=(1, 0)).detach().numpy()

    return translated_image

def translate_cube(cube,pattern_width,rate=0.6,direction='vertical'):
    m,n,_ = cube.shape
    if direction=='horizontal':
        nb_patterns = int(n/pattern_width)
        nb_displacement = int(np.floor(nb_patterns*rate))*pattern_width
        translated_image = torch.roll(torch.from_numpy(cube), shifts=(nb_displacement, 0), dims=(1, 0)).detach().numpy()

    if direction=='vertical':
        nb_patterns = int(m / pattern_width)
        nb_displacement = int(np.floor(nb_patterns*rate))*pattern_width
        translated_image = torch.roll(torch.from_numpy(cube), shifts=(0, nb_displacement),dims=(1, 0)).detach().numpy()
    if direction == 'vertical+horizontal':
        nb_patterns = int(m / pattern_width)
        nb_displacement = int(np.floor(nb_patterns*rate))*pattern_width
        translated_image = torch.roll(torch.from_numpy(cube), shifts=(nb_displacement, nb_displacement),
                                      dims=(1, 0)).detach().numpy()

    return translated_image


def shuffle_patterns(raw, sfa_size=(5,5)):
    tokens, coordinates = extract_raw_patches(raw, sfa_size)
    tokens = mix_tokens(tokens)
    ##coordinates = mix_tokens(coordinates)
    mixed_image = reshape_tokens(raw, tokens, coordinates, patch_size=sfa_size[0])
    return mixed_image

def shuffle_patterns_cube(cube, sfa_size=(5,5)):
    tokens, coordinates = extract_patches(cube, sfa_size)
    tokens = mix_tokens(tokens)
    ##coordinates = mix_tokens(coordinates)
    mixed_image = reshape_tokens(cube, tokens, coordinates, patch_size=sfa_size[0])
    return mixed_image


def raw_patches2cube(patches,coordinates,patch_size=5, image_size=(1020,1020)):
    raw_image = np.zeros((image_size[0], image_size[1]))
    counter = 0
    for cnt in range(len(coordinates)):
                raw_image[coordinates[cnt][0]:(coordinates[cnt][0] + patch_size),coordinates[cnt][1]:(coordinates[cnt][1] + patch_size)] = patches[counter]
                counter += 1
    return raw_image

def get_random_idx(max_samples, needed_samples, seed):
    random.seed(seed)  # in order to generate  always the same random idx for other tests when this function is called
    idx = random.sample(range(0, max_samples), needed_samples)
    return idx



def augment_train_raw(raw,sfa_size): # old tranformation used previously
    random.seed(42)
    x0 = HFlip(raw,sfa_size[0])
    x1 = shuffle_patterns(raw, sfa_size=(sfa_size[0]*7,sfa_size[1]*7))
    x2 = translate(raw, sfa_size[0], rate=0.6, direction='vertical')    #
    x3 = VFlip(raw,sfa_size[0])
    raw_augmented = [raw,x0,x1,x2,x3]
    return raw_augmented


def augment_train_raw_classic(raw): # classic tranformations applied to raw images
    random.seed(42)
    x0 = HFlip_2(raw)
    x2 = translate(raw, 3,rate=0.6, direction='vertical')    #
    x3 = VFlip_2(raw)
    raw_augmented = [raw,x0,x2,x3]
    return raw_augmented


def augment_train_demosaiced(cube,sfa_size):
    random.seed(42)
    x0 = HFlip_2(cube)
    x1 = shuffle_patterns_cube(cube, sfa_size=(sfa_size[0]*7,sfa_size[1]*7))
    x2 = translate_cube(cube, sfa_size[0], rate=0.6, direction='vertical')
    x3 = VFlip_2(cube)
    cube_augmented = [cube,x0,x1,x2,x3]
    return cube_augmented

def augment_test_raw(raw,sfa_size): # old tranformation used previously
    random.seed(42)
    x0 = add_noise_(raw, noise_std=0.25)
    x1 = translate(raw, sfa_size[0], rate=0.7, direction='horizontal')
    x2 = optical_distorsion(raw, sfa_size[0])
    raw_augmented = [raw,x0,x1,x2]
    return raw_augmented

def augment_test_demosaiced(cube,sfa_size):
    random.seed(42)
    x0 = add_noise_(cube, noise_std=0.25)
    x1 = translate_cube(cube, sfa_size[0], rate=0.7, direction='horizontal')
    aug = OpticalDistortion(p=1.0)
    augmented = aug(image=cube)
    x2 = augmented['image']
    cube_augmented = [cube,x0,x1,x2]
    return cube_augmented

def augment_test_raw_classic(raw): # old tranformation used previously
    random.seed(42)
    x0 = add_noise_(raw, noise_std=0.25)
    x1 = translate(raw, 3, rate=0.7, direction='horizontal')
    aug = OpticalDistortion(p=1.0)
    augmented = aug(image=raw)
    x2 = augmented['image']
    raw_augmented = [raw,x0,x1,x2]
    return raw_augmented



















