# Alternationg steepest descent for matrix completion MSFA demosaicing
import numpy as np
import numpy.linalg as la
import scipy.sparse as ssp
import scipy.sparse.linalg as sla

def to8Bit(I):
    mn = I.min()
    mx = I.max()
    mx-=mn
    I = ((I-mn)/mx)*255
    return I.astype(np.uint8)
def toFloat32(I):
    I = I/np.max(I)
    return I.astype(np.float32)


def get_H(pattern):
    NB_rows = pattern.shape[0]
    NB_cols = pattern.shape[1]
    vr = np.array([np.arange(1, NB_rows + 1).tolist() + np.arange(NB_rows - 1, 0, -1).tolist()]).T
    vc = np.array([np.arange(1, NB_cols + 1).tolist() + np.arange(NB_cols - 1, 0, -1).tolist()])
    return (vr * vc) / (vr.max() * vc.max())

def get_M(pattern): # coarse estimation
    NB_rows = pattern.shape[0]
    NB_cols = pattern.shape[1]
    M = np.ones((NB_rows, NB_cols))
    return M / M.sum()


def create_padded_sparse_image(mosaic_image,pattern_image,bands,kernel):
    padded_sparse_image = []
    padding_in_y = int(np.floor(kernel.shape[0] / 2))
    padding_in_x = int(np.floor(kernel.shape[1] / 2))
    for k in bands:
        tmp = mosaic_image * (pattern_image==k).astype(bool)
        padded_sparse_image.append(np.pad(tmp, [(padding_in_y, padding_in_x), (padding_in_y, padding_in_x)], mode="constant"))
    padded_mosaic_image = np.dstack(padded_sparse_image)
    return padded_mosaic_image

def create_sparse_image(mosaic_image,pattern_image):
    K_sparse_image = []
    bands = np.unique(pattern_image)
    for k in bands:
        tmp = mosaic_image * (pattern_image==k).astype(bool)
        K_sparse_image.append(tmp)
    K_sparse_image = np.dstack(K_sparse_image)
    return K_sparse_image

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


def get_WZ(X, rank):
    U, s, Vh = sla.svds(X, rank)
    for k in range(rank):
        U[:, k] = s[k] * U[:, k]
    return U, Vh


def apply_A(M, options):
    P = options['subsampling']
    return M.multiply(P)


def asd_demosaicing(raw_img,MSFA_pattern,
                GUESS=None, REFERENCE=None,
                rank=3, max_iter=20, tol_iter=1.0e-3):
    num_rows = raw_img.shape[0]
    num_cols = raw_img.shape[1]
    num_bands = len(np.unique(MSFA_pattern))
    pattern_image = create_pattern_image(MSFA_pattern, num_rows, num_cols)
    sparse_channels = (create_sparse_image(raw_img, pattern_image)).astype(float)
    mask = [((sparse_channels[:, :, i] != 0) * 1).astype('uint8') for i in range(num_bands)]
    mask = np.dstack(mask)  # make it a 3D array
    #
    sMSFA = ssp.csc_matrix(raw_img)
    options = {'subsampling': ssp.csc_matrix(mask.reshape((num_rows * num_cols, num_bands), order='F'))}

    Y = np.empty((num_rows * num_cols, num_bands))
    for k in range(num_bands):
        smaskk = ssp.csc_matrix(mask[:, :, k])
        # Y[:, k] = (sMSFA.multiply(smaskk)).reshape((num_rows * num_cols), order='F').toarray()
        Y[:, k] = (sMSFA.multiply(smaskk)).A.reshape((num_rows * num_cols), order='F')
    sY = ssp.csc_matrix(Y)
    norm_Y = sla.norm(sY)

    if (GUESS is not None):
        X = GUESS.reshape((num_rows * num_cols, num_bands), order='F')
    else:
        X = Y
    sX = ssp.csc_matrix(X)
    W, Z = get_WZ(sX, rank)
    sW = ssp.csc_matrix(W)
    sZ = ssp.csc_matrix(Z)
    sR = sY - apply_A(sX, options)
    err_res = sla.norm(sR) / norm_Y
    err_res_all = [err_res]
    err_true_all = None
    if (REFERENCE is not None):
        X_true = REFERENCE.reshape((num_rows * num_cols, num_bands), order='F')
        norm_X_true = la.norm(X_true)
        err_true = la.norm(X - X_true) / norm_X_true
        err_true_all = [err_true]
    l = 0
    while ((l < max_iter) & (err_res > tol_iter)):
        sGz = - sR @ sZ.T
        sGzZ = sGz @ sZ
        a = (sla.norm(sGz) / sla.norm(apply_A(sGzZ, options))) ** 2
        sW = sW - a * sGz
        sR = sR + a * apply_A(sGzZ, options)
        sGw = - sW.T @ sR
        sWGw = sW @ sGw
        b = (sla.norm(sGw) / sla.norm(apply_A(sWGw, options))) ** 2
        sZ = sZ - b * sGw
        sX = sW @ sZ
        sR = sY - apply_A(sX, options)
        err_res = sla.norm(sR) / norm_Y
        err_res_all.append(err_res)
        if (REFERENCE is not None):
            X = sX.toarray()
            err_true = la.norm(X - X_true) / norm_X_true
            err_true_all.append(err_true)
        l = l + 1
    IMAGE_ASD = X.reshape((num_rows, num_cols, num_bands), order='F')
    return IMAGE_ASD, err_res_all, err_true_all

