# -*- coding: utf-8 -*-
""" Title: Implementation of different deep models for crop/weed feature extraction.
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-06-2023
  """

import sys
sys.path.append('/../')
from types import SimpleNamespace
from descriptors import (vision_conformer, vit, spectral_former, resnet18, hsi_mixer, cvt, msfanet, mixformer, resmlp,
                         rawmixer, rawmixer_res, vgg11, cct)


def get_model(CONFIG):


    if CONFIG.model_name =='rawmixer_res':
        model = rawmixer_res.RawMixerRes(CONFIG.input_channels,
                                    CONFIG.msfa_width,
                256,
                320,
                2,
                2,
                CONFIG.logits_size,
                CONFIG.patch_size,
                dropout=0.0)

    elif CONFIG.model_name == 'VGG11':
        model = vgg11.VGG11(CONFIG.input_channels,CONFIG.patch_size[0],CONFIG.logits_size)

    elif CONFIG.model_name =='rawmixer':
        model = rawmixer.RawMixer(CONFIG.input_channels,
                                    CONFIG.msfa_width,
                256,
                384,
                2,
                2,
                CONFIG.logits_size,
                CONFIG.patch_size,
                dropout=0.0)

    elif CONFIG.model_name == 'Resmlp':
         model = resmlp.ResMLP(CONFIG.input_channels, 256, CONFIG.logits_size, CONFIG.msfa_width, CONFIG.patch_size[0], 2, 384)

    elif CONFIG.model_name == 'MixFormer':
        model = mixformer.MixFormer(128, 2, 384, CONFIG.token_width, CONFIG.patch_size[0], CONFIG.input_channels,CONFIG.logits_size,dropout=0.)

    elif CONFIG.model_name == 'MSFANet':
        model = msfanet.MsfaNet(CONFIG.input_channels,CONFIG.msfa_width, CONFIG.logits_size, use_maxpool = True, msfa_type = str(CONFIG.msfa_width)+'x'+str(CONFIG.msfa_width))
    elif CONFIG.model_name == 'ResNet18':
        model = resnet18.ResNet18(CONFIG.input_channels,256, CONFIG.logits_size)
    elif CONFIG.model_name == 'ViT':
        model = vit.VisionTransformer(256,384,2,2,CONFIG.logits_size,CONFIG.token_width,(CONFIG.input_channels,CONFIG.patch_size[0],CONFIG.patch_size[1]),dropout=0.0)
        # model = vit.VisionTransformer(256, 384, 4, 2, CONFIG.logits_size, 5,
        #                               (CONFIG.input_channels, CONFIG.patch_size[0], CONFIG.patch_size[1]), dropout=0.0)

    elif CONFIG.model_name == 'SpectralFormer_ViT':
        model = spectral_former.SpectralFormer(CONFIG.patch_size[0],CONFIG.token_width, CONFIG.input_channels,CONFIG.msfa_width, CONFIG.logits_size, 256, 2,
                                  2, 384, pool='cls' ,dim_head=16, dropout=0., emb_dropout=0., mode='ViT')
    elif CONFIG.model_name == 'SpectralFormer_CAF':
        # model = spectral_former.SpectralFormer(CONFIG.patch_size[0],CONFIG.token_width, CONFIG.input_channels,2, CONFIG.logits_size, 256, 2,
        #                           2, 384, pool='cls', dim_head=16, dropout=0., emb_dropout=0., mode='CAF')
        model = spectral_former.SpectralFormer(CONFIG.patch_size[0], CONFIG.token_width, CONFIG.input_channels, CONFIG.msfa_width,
                                               CONFIG.logits_size, 256, 2,
                                               2, 384, pool='cls', dim_head=16, dropout=0., emb_dropout=0., mode='CAF')

    elif CONFIG.model_name == 'Vision_Conformer':
        model = vision_conformer.VisionConformer(image_size=CONFIG.patch_size[0],
                                                patch_size = CONFIG.token_width,
                                                num_classes=CONFIG.logits_size,
                                                dim = 256,
                                                depth = 2,
                                                heads = 2,
                                                mlp_dim = 384,
                                                hidden_channels = 512,
                                                cnn_depth = 2,
                                                pool = 'cls',
                                                channels = CONFIG.input_channels,
                                                dim_head = 64,
                                                dropout = 0.,
                                                emb_dropout = 0.)
    elif CONFIG.model_name == 'CvT':
        kwargs = cvt.kwargs
        kwargs['num_classes'] = CONFIG.logits_size
        model = cvt.ConvolutionalVisionTransformer(kwargs)

    elif CONFIG.model_name == 'CCT':
        model = cct._cct(CONFIG.patch_size[0],  CONFIG.input_channels, 2, 2, 1.5, 256, num_classes=CONFIG.logits_size, kernel_size=3, stride=None, padding=None)

    elif CONFIG.model_name == 'hsi_mixer':
        model = hsi_mixer.HSI_Mixer_Net(num_classes=CONFIG.logits_size, img_size=CONFIG.patch_size[0], patch_size=7, depth=1, in_chans=CONFIG.input_channels, embed_dim=768)
    else:
        print('Model not implemented')
    return model


def get_nb_params(model_names,logits_size=2,patch_size=25,input_channels=18):
    model_parameters = []
    for arch in model_names:
        args = SimpleNamespace(logits_size=logits_size,num_classes=2, model_name=arch, input_channels=input_channels, patch_size=(patch_size, patch_size))
        model = get_model(args)
        model_param = (str(arch)+' : '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        model_parameters.append(model_param)

    return model_parameters



