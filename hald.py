import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import pdb

def trilinear_interpolate(hald_img, clut_r, clut_g, clut_b, clut_size):
    print("clut_size: ", clut_size)
    r0 = np.floor(clut_r).astype(int)
    r1 = np.minimum(r0 + 1, clut_size - 1).astype(int)
    g0 = np.floor(clut_g).astype(int)
    g1 = np.minimum(g0 + 1, clut_size - 1).astype(int)
    b0 = np.floor(clut_b).astype(int)
    b1 = np.minimum(b0 + 1, clut_size - 1).astype(int)

    r_ratio = (clut_r - r0)[:,:,np.newaxis] # calculate fractional parts for Red channel
    g_ratio = (clut_g - g0)[:,:,np.newaxis] # calculate fractional parts for Green channel
    b_ratio = (clut_b - b0)[:,:,np.newaxis] # calculate fractional parts for Blue channel
    r_ratio = b_ratio = g_ratio

    c000 = hald_img[r0 + clut_size * (g0 + clut_size * b0)]
    c100 = hald_img[r1 + clut_size * (g0 + clut_size * b0)]
    c010 = hald_img[r0 + clut_size * (g1 + clut_size * b0)]
    c110 = hald_img[r1 + clut_size * (g1 + clut_size * b0)]
    c001 = hald_img[r0 + clut_size * (g0 + clut_size * b1)]
    c101 = hald_img[r1 + clut_size * (g0 + clut_size * b1)]
    c011 = hald_img[r0 + clut_size * (g1 + clut_size * b1)]
    c111 = hald_img[r1 + clut_size * (g1 + clut_size * b1)]

    # interpolate alongside red channel
    c00 = c000 * (1 - r_ratio) + c100 * r_ratio
    c01 = c001 * (1 - r_ratio) + c101 * r_ratio
    c10 = c010 * (1 - r_ratio) + c110 * r_ratio
    c11 = c011 * (1 - r_ratio) + c111 * r_ratio

    # interpolate alongside green channel
    c0 = c00 * (1 - g_ratio) + c10 * g_ratio
    c1 = c01 * (1 - g_ratio) + c11 * g_ratio

    # interpolate alongside blue channel
    final_color = c0 * (1 - b_ratio) + c1 * b_ratio

    # return np.clip(final_color, 0, 255)
    return final_color


def apply_hald_clut(hald_img, img):
    hald_w, hald_h, channels = hald_img.shape
    clut_size = int(round(math.pow(hald_w*hald_h, 1/3)))
    scale = (clut_size - 1) / 255

    img = np.asarray(img).astype(float)
    hald_img = np.asarray(hald_img).astype(float)

    print("hald image before reshaped.shape:", hald_img.shape)
    hald_img = hald_img.reshape(clut_size ** 3, channels)
    print("hald_img_reshaped.shape:", hald_img.shape)
     
    # Figure out the 3D CLUT indexes corresponding to the pixels in our image
    # Normalize and scale RGB values to the range [0, clutSize - 1)
    clut_r = img[:, :, 0] * scale
    clut_g = img[:, :, 1] * scale
    clut_b = img[:, :, 2] * scale

    # filtered_image = trilinear_interpolate(hald_img, clut_r, clut_g, clut_b, clut_size)

    ####### Constructed not interpolated filtered_image_int for debug purposes
    clut_r_int = np.rint(clut_r).astype(int)
    clut_g_int = np.rint(clut_g).astype(int)
    clut_b_int = np.rint(clut_b).astype(int)

    # # Clamping indices to prevent out-of-bounds indexing
    # clut_r_int = np.clip(clut_r_int, 0, clut_size - 1)
    # clut_g_int = np.clip(clut_g_int, 0, clut_size - 1)
    # clut_b_int = np.clip(clut_b_int, 0, clut_size - 1)

    indices = clut_r_int + clut_size * (clut_g_int + clut_size * clut_b_int)

    filtered_image_int = np.zeros((img.shape))
    filtered_image_int[:, :] = hald_img[np.rint(indices).astype(int)]
    cv2.imwrite("filtered_int.png", filtered_image_int)

    ####### DEBUGING END

    return filtered_image_int.astype(np.uint8)


hald_img = cv2.imread("dehancer-fujichrome-velvia-50-k2383.png", cv2.IMREAD_UNCHANGED)  # Reads with alpha channel
hald = hald_img  # Drops the alpha channel
hald = cv2.cvtColor(hald, cv2.COLOR_BGR2RGB)

image = cv2.imread("hald_8.png") # genered with imagemagick "convert hald:8 -depth 8 -colorspace sRGB hald_8.png"

filtered = apply_hald_clut(hald,image)
print("diff of original dehancer LUT and my generated LUT", np.mean(hald - filtered))
cv2.imwrite("filtered.png", filtered)
