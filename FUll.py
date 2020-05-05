import os
import numpy as np
import scipy
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt
import collections
import random as rand



def patchify(img, patch_shape):
    X, Y = img.shape
    x, y = patch_shape
    shape = (X - x + 1, Y - y + 1, x, y)
    X_str, Y_str= img.strides
    strides = (X_str, Y_str, X_str, Y_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


if __name__== '__main__':
    file_name =os.path.join('lena_color_512.tif')
    img =rgb2gray(plt.imread(file_name))
    
    noisy = add_gaussian_noise(img, sigma = 20)
    
    kernel =gaussian_kernel(3)
    filtered= wiener_filter(noisy,kernel,K=10)
    plt.imshow(noisy,cmap='gray')
    
    
    noisy1=noisy
    fil=filtered
    
    patches_noisy=patchify(noisy1,(11,11))
    
    patches_fil=patchify(fil,(11,11))
    
    
    