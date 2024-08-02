import math
import numpy as np
import cv2

def clamp(value, min_value, max_value):
    return np.clip(value, min_value, max_value)

def trilinear_interpolate(hald_img, clut_r, clut_g, clut_b, clut_size):
    r0 = np.floor(clut_r).astype(int)
    r1 = np.ceil(clut_r).astype(int)
    g0 = np.floor(clut_g).astype(int)
    g1 = np.ceil(clut_g).astype(int)
    b0 = np.floor(clut_b).astype(int)
    b1 = np.ceil(clut_b).astype(int)

    r_ratio = (clut_r - r0)[:, :, np.newaxis]
    g_ratio = (clut_g - g0)[:, :, np.newaxis]
    b_ratio = (clut_b - b0)[:, :, np.newaxis]

    # np.savetxt("foo.csv", r_ratio.reshape(r_ratio.shape[0], -1), delimiter=",")
    
    c000 = np.clip(hald_img[r0 + clut_size ** 2 * g0 + clut_size ** 4 * b0], 0, 255)
    c001 = np.clip(hald_img[r0 + clut_size ** 2 * g0 + clut_size ** 4 * b1], 0, 255)
    c010 = np.clip(hald_img[r0 + clut_size ** 2 * g1 + clut_size ** 4 * b0], 0, 255)
    c011 = np.clip(hald_img[r0 + clut_size ** 2 * g1 + clut_size ** 4 * b1], 0, 255)

    c100 = np.clip(hald_img[r1 + clut_size ** 2 * g0 + clut_size ** 4 * b0], 0, 255)
    c101 = np.clip(hald_img[r1 + clut_size ** 2 * g0 + clut_size ** 4 * b1], 0, 255)
    c110 = np.clip(hald_img[r1 + clut_size ** 2 * g1 + clut_size ** 4 * b0], 0, 255)
    c111 = np.clip(hald_img[r1 + clut_size ** 2 * g1 + clut_size ** 4 * b1], 0, 255)
    # cv2.imwrite("c000.jpg", c000)
    # cv2.imwrite("c001.jpg", c001)
    # cv2.imwrite("c010.jpg", c010)
    # cv2.imwrite("c011.jpg", c011)
    # cv2.imwrite("c100.jpg", c100)
    # cv2.imwrite("c101.jpg", c101)
    # cv2.imwrite("c110.jpg", c110)
    # cv2.imwrite("c111.jpg", c111)

    print(c000.shape, r_ratio.shape, c100.shape)
    c00 = np.clip((c000 * (1 - r_ratio) + c100 * r_ratio), 0, 255)
    c01 = np.clip((c001 * (1 - r_ratio) + c101 * r_ratio), 0, 255)
    c10 = np.clip((c010 * (1 - r_ratio) + c110 * r_ratio), 0, 255)
    c11 = np.clip((c011 * (1 - r_ratio) + c111 * r_ratio), 0, 255)
    # cv2.imwrite("c00.jpg", c00)
    # cv2.imwrite("c01.jpg", c01)
    # cv2.imwrite("c10.jpg", c10)
    # cv2.imwrite("c11.jpg", c11)

    c0 = np.clip((c00 * (1 - g_ratio) + c10 * g_ratio), 0, 255)
    c1 = np.clip((c01 * (1 - g_ratio) + c11 * g_ratio), 0, 255)
    # cv2.imwrite("c0.jpg", c00)
    # cv2.imwrite("c1.jpg", c01)


    print(c0.shape, b_ratio.shape, c1.shape, b_ratio.shape)
    filtered_image = np.clip((c0 * (1 - b_ratio) + c1 * b_ratio), 0, 255)

    return filtered_image
    # return filtered_image.astype(np.uint8)


def apply_hald_clut(hald_img, img):
    hald_img = cv2.cvtColor(hald_img, cv2.COLOR_BGR2RGB)
    hald_w, hald_h, channels = hald_img.shape

    clut_size = int(round(math.pow(hald_w, 1/3)))
    scale = (clut_size * clut_size - 1) / 255

    img = np.asarray(img).astype(float)
    hald_img = np.asarray(hald_img)
    hald_img = hald_img.reshape(clut_size ** 6, 3)
    
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


hald = cv2.imread("/home/rafael/.local/bin/luts/dehancer-fujichrome-velvia-50-k2383.png")
# image = cv2.imread("/home/rafael/phone/DCIM/OpenCamera/IMG_20240727_115819.jpg")
image = cv2.imread("orig_orig.jpg")
filtered = apply_hald_clut(hald,image)
cv2.imwrite("filtered.jpg", filtered)
# cv2.imwrite("orig.jpg", image)
