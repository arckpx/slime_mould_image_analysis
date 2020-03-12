import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd


def remove_dish(img):
    kernel = np.ones((5, 5), np.uint8)
    _, binimg = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, kernel)
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, kernel)
    dish = np.where(0 < binimg)
    cx = (dish[1][-1] + dish[1][0])/2
    cy = (dish[0][-1] + dish[0][0])/2
    r = np.sqrt((dish[1][-1]-dish[1][0])**2 + (dish[0][-1]-dish[0][0])**2) / 2

    shrink = 40
    ny, nx = np.shape(img)
    y, x = np.ogrid[-cy:ny-cy, -cx:nx-cx]
    mask = x*x + y*y <= (r-shrink)*(r-shrink)
    mask.dtype = 'uint8'

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img


def detect_object(img, thres):
    _, binimg = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3))
    img_lbl, num_spots = nd.label(binimg, structure=kernel)
    counts = np.bincount(np.ndarray.flatten(img_lbl))
    blob_lbl = counts[1:].argmax()+1
    obj = (img_lbl == blob_lbl)
    obj.dtype = 'uint8'
    return obj


def calc_area(slime):
    return np.sum(slime)


def calc_perimeter(slime):
    closed_slime = nd.binary_fill_holes(slime).astype('uint8')
    kernel = np.ones((3, 3))
    erosion = cv2.erode(closed_slime, kernel)
    closed_slime -= erosion
    return np.sum(closed_slime)


def calc_circularity(area, perimeter):
    return (4*np.pi*area)/(perimeter*perimeter)


def centroid(obj):
    m = cv2.moments(obj)
    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])
    return np.array([cx, cy])


def calc_reach_distance(oat, slime):
    sy, sx = np.where(slime)
    slime_pos = np.transpose([sx, sy])
    dist_array = np.linalg.norm(slime_pos - centroid(oat), axis=1)
    return max(dist_array)


def calc_move_distance(oat, slime):
    ix, iy = centroid(oat)
    fx, fy = centroid(slime)
    return np.sqrt((fx-ix)**2 + (fy-iy)**2)


def calc_flatness(slime):
    y, x = np.where(slime)
    c = np.cov(x, y)
    eigval, eigvec = np.linalg.eig(c)
    return 1-abs(min(eigval)/max(eigval))


def plot_cov_eig(slime, scale=300):
    y, x = np.where(slime)
    c = np.cov(x, y)
    eigval, eigvec = np.linalg.eig(c)
    eigval /= max(eigval)
    eigval *= scale
    eigvec *= eigval

    origin = centroid(slime)
    e1 = (eigvec[:, 0]).astype(int)
    e2 = (eigvec[:, 1]).astype(int)
    x1, y1 = np.transpose([origin-e1, origin+e1])
    x2, y2 = np.transpose([origin-e2, origin+e2])

    plt.figure()
    plt.title('Covariance Matrix Eigenvectors')
    plt.imshow(slime, cmap='gray')
    plt.plot(x1, y1, '-m', linewidth=2)
    plt.plot(x2, y2, '-m', linewidth=2)
    plt.show()


def main():
    img_name = 'sample_image.tif'
    img = cv2.imread(img_name, 0)
    img = remove_dish(img)

    thres_slime = 43
    thres_oat = 100
    slime = detect_object(img, thres_slime)
    oat = detect_object(img, thres_oat)

    area = calc_area(slime)
    perimeter = calc_perimeter(slime)
    circularity = calc_circularity(area, perimeter)
    reach_dist = calc_reach_distance(oat, slime)
    move_dist = calc_move_distance(oat, slime)
    flatness = calc_flatness(slime)

    print('AREA:           {}'.format(area))
    print('PERIMETER:      {}'.format(perimeter))
    print('CIRCULARITY:    {:.5f}'.format(circularity))
    print('REACH DISTANCE: {:.3f}'.format(reach_dist))
    print('MOVE DISTANCE:  {:.3f}'.format(move_dist))
    print('FLATNESS:       {:.5f}'.format(flatness))
    plot_cov_eig(slime)


if __name__ == '__main__':
    main()
