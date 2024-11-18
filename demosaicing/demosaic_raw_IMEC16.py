import glob,time
import numpy as np
import spectral.io.envi as envi
import os, sys
sys.path.append('/../')
from utils import mosaic_tools as mt
from demosaicing import asd,sd
from tqdm import tqdm


def to8Bit(I):
    mn = I.min()
    mx = I.max()
    mx-=mn
    I = ((I-mn)/mx)*255
    return I.astype(np.uint8)
def toFloat32(I):
    I = I/np.max(I)
    return I.astype(np.float32)

def get_hdr_and_raw_paths(images_path):
    contain_hdrs = glob.iglob(images_path + '**/*.hdr', recursive=True)
    contain_raws = glob.iglob(images_path + '**/*.raw', recursive=True)
    hdrs = []
    raws = []
    imageLists = []
    for f1 in contain_hdrs:
        hdrs.append(f1)
    for f2 in contain_raws:
        raws.append(f2)
    for i in range(len(hdrs)):
        filename = os.path.basename(hdrs[i])
        name, extension = os.path.splitext(filename)
        for ii in range(len(raws)):
            filename2 = os.path.basename(raws[ii])
            name2, extension2 = os.path.splitext(filename2)
            if name == name2:
                temp = (hdrs[i], raws[ii])
                imageLists.append(temp)
    return imageLists
#
images_path = '/path to IMEC 4x4 raw dataset/'
target_path = '//'
imageLists = get_hdr_and_raw_paths(images_path)
MSFA_pattern = mt.make_pattern(pattern_type='Imec16')
demosaicing_times = []
for i in tqdm(range(len(imageLists))):
    img_path = imageLists[i]
    obj = envi.open(img_path[0], img_path[1])
    directory = os.path.dirname(obj.filename)  # to check if patch_coordinates.txt exists
    filename = os.path.basename(obj.filename)
    name, extension = os.path.splitext(filename)
    if not os.path.exists(target_path + name):
        os.mkdir(target_path + name)
    folder_path = target_path + name + '/'
    output_path_final = folder_path + name + '.hdr'
    raw = obj.asarray()[:,:,0]
    #
    start_time = time.time()
    sd_demosaiced = sd.sd_demosaicing(raw, MSFA_pattern,IMAGE_WB=None)
    asd_demosaiced, err_res_all, err_true_all = asd.asd_demosaicing(raw.astype(np.float32),MSFA_pattern,
                 GUESS=sd_demosaiced.astype(np.float32), REFERENCE=None,
                 rank=3, max_iter=20, tol_iter=1.0e-3)
    end_time = time.time()
    demosaicing_times.append(end_time - start_time)
    #
    asd_demosaiced = to8Bit(asd_demosaiced)
    #
    metadata = obj.metadata
    envi.save_image(output_path_final, asd_demosaiced, dtype=np.uint8,interleave='bsq', ext='.raw', metadata=metadata, force=True)