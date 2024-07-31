import math 
import numpy as np
import cv2
def apply_hald_clut(hald_img, img):
    hald_img = cv2.cvtColor(hald_img, cv2.COLOR_BGR2RGB)
    hald_w, hald_h, channels = hald_img.shape
    clut_size = int(round(math.pow(hald_w, 1/3)))
    scale = (clut_size * clut_size - 1) / 255
    img = np.asarray(img)
    hald_img = np.asarray(hald_img)
    hald_img = hald_img.reshape(clut_size ** 6, 3)
    
    # Figure out the 3D CLUT indexes corresponding to the pixels in our image
    clut_r = np.rint(img[:, :, 0] * scale).astype(int)
    clut_g = np.rint(img[:, :, 1] * scale).astype(int)
    clut_b = np.rint(img[:, :, 2] * scale).astype(int)

    filtered_image = np.zeros((img.shape))
    filtered_image = filtered_image.astype(np.uint8)
    filtered_image[:, :] = hald_img[clut_r + clut_size ** 2 * clut_g + clut_size ** 4 * clut_b]
    return filtered_image


hald = cv2.imread("/home/rafael/.local/bin/luts/dehancer-fujichrome-velvia-50-k2383.png")
image = cv2.imread("/home/rafael/phone/DCIM/OpenCamera/IMG_20240727_115819.jpg")
filtered = apply_hald_clut(hald,image)
cv2.imwrite("filtered.jpg", filtered)
# cv2.imwrite("orig.jpg", image)
