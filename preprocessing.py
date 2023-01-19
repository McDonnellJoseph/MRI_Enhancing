import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms  # hop

# paths = [f'Project_MLSP/Data/sub-TELL01{str(i).zfill(2)}_T1w_brain_homog_mni.nii.gz' for i in range(1,11)]
def load(paths):
    """Just loads it

    Parameters
    ----------
    paths : list
        paths to the .nii.gz files

    Returns
    -------
    array of shape (n_files, dim_1, dim_2, dim_3)
        loaded images
    """
    return np.array([nib.load(path).get_fdata() for path in paths])

def normalize(raw_imgs):
    """Simple normalization

    Parameters
    ----------
    raw_imgs : array
        4 dimensions (first one being the different images)

    Returns
    -------
    array (same shape as raw_imgs)
        normalized images
    """
    return (raw_imgs - raw_imgs.min()) / (raw_imgs.max() - raw_imgs.min())

def simulate_3T(imgs, sigma=0.05):
    """ Adds noise to get 3T

    Parameters
    ----------
    imgs : array
        normalized images
    sigma : float, optional
        white noise parameter, by default 0.05

    Returns
    -------
    array (same shape as imgs)
        normalized images
    """
    return imgs + np.random.normal(0, sigma, size=imgs.shape)

def apply_hist_matching(imgs):
    """ Applies histogram matching

    Parameters
    ----------
    imgs : array
        normalized images

    Returns
    -------
    array (same shape as imgs)
        normalized images
    """
    hist_matched_imgs = imgs.copy()
    for i in range(1, len(imgs)):
        hist_matched_imgs[i] = match_histograms(imgs[i], imgs[0])
    return hist_matched_imgs

def preprocess(paths):
    """Normalize, simulate 3T and apply histogram matching to 7T images

    Parameters
    ----------
    paths : list
        paths to the .nii.gz files

    Returns
    -------
    tuple
        first element corresponds to preprocessed (simulated) 3T images, and second one corresponds to preprocessed 7T images
    """
    raw_imgs_7T = load(paths=paths)
    imgs_7T = normalize(raw_imgs=raw_imgs_7T)
    imgs_3T = simulate_3T(imgs_7T)
    imgs_7T = apply_hist_matching(imgs_7T)
    imgs_3T = apply_hist_matching(imgs_3T)
    return (imgs_3T, imgs_7T)

