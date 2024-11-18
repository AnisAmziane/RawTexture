# -*- coding: utf-8 -*-
""" Title: Convert HyTexila reflectance to radiance under extended A illuminant
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
#*******************************************************
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

HyTexila_centers =  np.asarray([405.369888,  408.561553,  411.753217,  414.944882,  418.136547, 421.328212,  424.519877,  427.711541,  430.903206,  434.094871,
        437.286536,  440.478201,  443.669865,  446.86153 ,  450.053195, 453.24486 ,  456.436524,  459.628189,  462.819854,  466.011519,
        469.203184,  472.394848,  475.586513,  478.778178,  481.969843, 485.161507,  488.353172,  491.544837,  494.736502,  497.928167,
        501.119831,  504.311496,  507.503161,  510.694826,  513.886491, 517.078155,  520.26982 ,  523.461485,  526.65315 ,  529.844814,
        533.036479,  536.228144,  539.419809,  542.611474,  545.803138, 548.994803,  552.186468,  555.378133,  558.569797,  561.761462,
        564.953127,  568.144792,  571.336457,  574.528121,  577.719786, 580.911451,  584.103116,  587.294781,  590.486445,  593.67811 ,
        596.869775,  600.06144 ,  603.253104,  606.444769,  609.636434, 612.828099,  616.019764,  619.211428,  622.403093,  625.594758,
        628.786423,  631.978087,  635.169752,  638.361417,  641.553082, 644.744747,  647.936411,  651.128076,  654.319741,  657.511406,
        660.70307 ,  663.894735,  667.0864,  670.278065,  673.46973 , 676.661394,  679.853059,  683.044724,  686.236389,689.428054,
        692.619718,  695.811383,  699.003048,  702.194713,  705.386377, 708.578042,  711.769707,  714.961372,  718.153037,  721.344701,
        724.536366,  727.728031,  730.919696,  734.11136 ,  737.303025, 740.49469 ,  743.686355,  746.87802 ,  750.069684,  753.261349,
        756.453014,  759.644679,  762.836344,  766.028008,  769.219673, 772.411338,  775.603003,  778.794667,  781.986332,  785.177997,
        788.369662,  791.561327,  794.752991,  797.944656,  801.136321, 804.327986,  807.51965 ,  810.711315,  813.90298 ,  817.094645,
        820.28631 ,  823.477974,  826.669639,  829.861304,  833.052969, 836.244634,  839.436298,  842.627963,  845.819628,  849.011293,
        852.202957,  855.394622,  858.586287,  861.777952,  864.969617, 868.161281,  871.352946,  874.544611,  877.736276,  880.92794 ,
        884.119605,  887.31127 ,  890.502935,  893.6946,  896.886264, 900.077929,  903.269594,  906.461259,  909.652923,  912.844588,
        916.036253,  919.227918,  922.419583,  925.611247,  928.802912, 931.994577,  935.186242,  938.377907,  941.569571,  944.761236,
        947.952901,  951.144566,  954.33623 ,  957.527895,  960.71956 , 963.911225,  967.10289 ,  970.294554,  973.486219,  976.677884,
        979.869549,  983.061213,  986.252878,  989.444543,  992.636208,995.827873])

### ------------------------------------------------------------------
source_path = 'path to dataset'
destination_path = ''
imageLists = get_hdr_and_raw_paths(source_path)
### ------------------------------------------------------------------
illuminant_A = np.load('extended_A_380_1000_interp1nm.npy')
_, idx_illu_for_hytexila = ST.find_closest_spectra_illuminant(HyTexila_centers,illuminant_A[:,0])
selected_A_values = illuminant_A[idx_illu_for_hytexila,1]
selected_A_values = selected_A_values.astype(np.float32)
### ------------------------------------------------------------------
for i in tqdm(range(len(imageLists))):
    img_path = imageLists[i]
    obj = envi.open(img_path[0], img_path[1])
    filename = os.path.basename(obj.filename)
    name, extension = os.path.splitext(filename)
    if not os.path.exists(destination_path + name):
        os.mkdir(destination_path + name)
    folder_path = destination_path + name + '/'
    output_path_final = folder_path + name + '.hdr'
    #
    cube = obj.asarray()
    cube = to_float32(cube)
    radiance_cube = ST.reflectance2radiance(cube, selected_A_values)
    radiance_cube = to_uint8(radiance_cube)
    envi.save_image(output_path_final, radiance_cube, dtype=np.uint8,interleave='bsq', ext='.raw', metadata=obj.metadata, force=True)
