# -*- coding: utf-8 -*-
""" Title: Convert HyTexila reflectance to radiance under extended Solar illuminant
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2024
  """
import sys
sys.path.append('../')
from utils import SpectralTransform as ST
import numpy as np
import spectral.io.envi as envi
import glob,os
from tqdm import tqdm

def t_tif_read(filename, bands=None):
    """
    A TIFF reader for spectral image cubes.

    Parameters:
    filename : str
        The input TIFF file to be read.
    bands : list of int, optional
        A list specifying which bands to load. Bands are counted starting from 1.
        If left empty, all bands will be loaded.
        If an invalid value is given, only the thumbnail image will be loaded, if it exists.

    Returns:
    im : ndarray
        The output image cube.
    tumb : ndarray
        The thumbnail image, if it exists.
    ifd : list
        The image file directory (IFD) of the input TIFF file.
    tif : list
        The parsed IFD.
    """
    if bands is None:
        bands = []

    # Read the TIFF file
    with tiff.TiffFile(filename) as tif:
        ifds = tif.pages
        num_ifds = len(ifds)

        # Initialize output variables
        im = []
        tumb = None
        ifd = []
        tif_parsed = []

        # Read IFDs
        for i, page in enumerate(ifds):
            ifd.append(page.tags)
            tif_parsed.append(page.asarray())

        # Check for thumbnail
        ifd_first = ifd[0]
        first_page = tif.pages[0]
        if first_page.photometric == tiff.PHOTOMETRIC.RGB and first_page.samplesperpixel == 3 and first_page.bitspersample == 8:
            tumb = first_page.asarray()
            flag_tumb = True
        else:
            flag_tumb = False

        # Load specified bands
        if not bands:
            bands = list(range(1, num_ifds))

        bands = [b for b in bands if 1 <= b <= num_ifds]

        if flag_tumb:
            bands = [b + 1 for b in bands]
            bands = list(set(bands))
            im = np.zeros((tif.pages[1].imagelength, tif.pages[1].imagewidth, len(bands)), dtype=tif.pages[1].dtype)
            if not bands:
                bands = [1]
        else:
            bands = list(set(bands))
            im = np.zeros((tif.pages[0].imagelength, tif.pages[0].imagewidth, len(bands)), dtype=tif.pages[0].dtype)

        c_count = 0
        for i, page in enumerate(tif.pages):
            if i + 1 in bands:
                data = page.asarray()
                if i == 0 and flag_tumb:
                    tumb = data
                else:
                    im[..., c_count] = data
                    c_count += 1

    return im, tumb, ifd, tif_parsed

def to_float32(I):
  MAX = np.max(I)
  I = I/MAX
  return I.astype('float32')

def to_float16(I):
  MAX = np.max(I)
  I = I/MAX
  return I.astype('float16')

def to_uint8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn) / mx) * 255
    return I.astype(np.uint8)

Spectex_centers =  np.asarray([400,405,410,415,420,425,430,435,440,445,450,455, 460,465,470,475,480,485,490,495,
 500,505,510, 515,520,
 525,
 530,
 535,
 540,
 545,
 550,
 555,
 560,
 565,
 570,
 575,
 580,
 585,
 590,
 595,
 600,
 605,
 610,
 615,
 620,
 625,
 630,
 635,
 640,
 645,
 650,
 655,
 660,
 665,
 670,
 675,
 680,
 685,
 690,
 695,
 700,
 705,
 710,
 715,
 720,
 725,
 730,
 735,
 740,
 745,
 750,
 755,
 760,
 765,
 770,
 775,
 780])

### ------------------------------------------------------------------
source_path = '/home/anis/Documents/ms_base/others/SpecTex/SpecTex_cubes_5nm/'
destination_path = '/home/anis/Documents/ms_base/others/SpecTex/Spectex_D65_uint8/'
imageLists = os.listdir(source_path)
### ------------------------------------------------------------------
illuminant_Solar = np.load('/.../utils/extended_A_380_1000_interp1nm.npy')
_, idx_illu_for_spectex = ST.find_closest_spectra_illuminant(Spectex_centers,illuminant_Solar[:,0])
selected_Solar_values = illuminant_Solar[idx_illu_for_spectex,1]
selected_Solar_values = selected_Solar_values.astype(np.float32)
### ------------------------------------------------------------------
for i in tqdm(range(len(imageLists))):
    img_path = imageLists[i]
    obj = source_path+imageLists[i]
    filename = os.path.basename(obj)
    name, extension = os.path.splitext(filename)
    if not os.path.exists(destination_path + name):
        os.mkdir(destination_path + name)
    folder_path = destination_path + name + '/'
    output_path_final = folder_path + name + '.hdr'
    #
    cube = t_tif_read(obj, bands=None)[0]
    cube = to_float32(cube)
    radiance_cube = ST.reflectance2radiance(cube, selected_Solar_values)
    radiance_cube = to_uint8(radiance_cube)
    metadata = {"wavelength":Spectex_centers,
    "samples" : cube.shape[0],
    "lines" : cube.shape[1],
    "bands" : cube.shape[2],}
    envi.save_image(output_path_final, radiance_cube, dtype=np.uint8,interleave='bsq', ext='.raw', metadata=metadata, force=True)