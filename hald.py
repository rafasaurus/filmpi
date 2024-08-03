import math
import numpy as np
import cv2

def trilinear_interpolate(hald_img, clut_r, clut_g, clut_b, clut_size):
    r0 = np.floor(clut_r).astype(int)
    r1 = np.minimum(r0 + 1, clut_size - 1).astype(int)
    g0 = np.floor(clut_g).astype(int)
    g1 = np.minimum(g0 + 1, clut_size - 1).astype(int)
    b0 = np.floor(clut_b).astype(int)
    b1 = np.minimum(b0 + 1, clut_size - 1).astype(int)

    r_ratio = (clut_r - r0)[:, :, np.newaxis]
    g_ratio = (clut_g - g0)[:, :, np.newaxis]
    b_ratio = (clut_b - b0)[:, :, np.newaxis]

    c000 = hald_img[r0 + clut_size * (g0 + clut_size * b0)]
    c100 = hald_img[r1 + clut_size * (g0 + clut_size * b0)]
    c010 = hald_img[r0 + clut_size * (g1 + clut_size * b0)]
    c110 = hald_img[r1 + clut_size * (g1 + clut_size * b0)]
    c001 = hald_img[r0 + clut_size * (g0 + clut_size * b1)]
    c101 = hald_img[r1 + clut_size * (g0 + clut_size * b1)]
    c011 = hald_img[r0 + clut_size * (g1 + clut_size * b1)]
    c111 = hald_img[r1 + clut_size * (g1 + clut_size * b1)]

    c00 = c000 * (1 - r_ratio) + c100 * r_ratio
    c01 = c001 * (1 - r_ratio) + c101 * r_ratio
    c10 = c010 * (1 - r_ratio) + c110 * r_ratio
    c11 = c011 * (1 - r_ratio) + c111 * r_ratio

    c0 = c00 * (1 - g_ratio) + c10 * g_ratio
    c1 = c01 * (1 - g_ratio) + c11 * g_ratio

    final_color = c0 * (1 - b_ratio) + c1 * b_ratio
    return np.clip(final_color, 0, 255)


def apply_hald_clut(hald_img, img):
    hald_img = cv2.cvtColor(hald_img, cv2.COLOR_BGR2RGB)
    hald_w, hald_h, channels = hald_img.shape

    clut_size = int(round(math.pow(hald_w*hald_h, 1/3)))
    scale = (clut_size - 1) / 255

    img = np.asarray(img).astype(float)
    hald_img = np.asarray(hald_img)
    hald_img = hald_img.reshape(clut_size ** 3, 3)
    
    # Figure out the 3D CLUT indexes corresponding to the pixels in our image
    # Normalize and scale RGB values to the range [0, clutSize - 1)
    clut_r = img[:, :, 0].astype(float) * scale
    clut_g = img[:, :, 1].astype(float) * scale
    clut_b = img[:, :, 2].astype(float) * scale

    # filtered_image = np.zeros((img.shape), np.uint8)

    cv2.imwrite("r.jpg", clut_r)
    cv2.imwrite("g.jpg", clut_g)
    cv2.imwrite("b.jpg", clut_b)
    # filtered_image[:, :] = hald_img[clut_r +
    #                                 clut_size ** 2 * clut_g +
    #                                 clut_size ** 4 * clut_b]
    filtered_image = trilinear_interpolate(hald_img, clut_r, clut_g, clut_b, clut_size)
    return filtered_image


# hald = cv2.imread("dehancer-fujichrome-velvia-50-k2383.png")
hald_img = cv2.imread("dehancer-fujichrome-velvia-50-k2383.png", cv2.IMREAD_UNCHANGED)  # Reads with alpha channel
hald = hald_img[:, :, :3]  # Drops the alpha channel
image = cv2.imread("orig_orig.jpg")

filtered = apply_hald_clut(hald,image)
cv2.imwrite("filtered.jpg", filtered)
# cv2.imwrite("orig.jpg", image)
