import numpy as np
import os
import morphsnakes as ms
import sys
from morphsnakes import (
    morphological_geodesic_active_contour, inverse_gaussian_gradient,
    circle_level_set)
import matplotlib
from matplotlib import pyplot as plt
import time


def save_img(img, ls, name):
    fig, ax = plt.subplots()

    ax.imshow(img)
    ax.set_axis_off()
    ax.contour(ls, [0.5], colors='r')

    fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fig.savefig(dir_path + '/tmp/' + name, pad_inches=0, bbox_inches='tight')
    plt.close(fig)


def middle_of_line(row):
    start = 0
    end = 0
    started = False
    i = 0
    for x in row:
        if x == 1 and started == False:
            started = True
            start = i
        if x == 0 and started == True:
            end = i
            break
        i += 1
    return int((end + start) / 2)

def acwe3d(img, coord, iterations, smoothing):
    print('Running: snake_3d (MorphACWE)...')

    init_ls = ms.circle_level_set(img.shape, (coord[2], coord[1], coord[0]), 5)

    ls = ms.morphological_chan_vese(img, iterations=iterations,
                                    init_level_set=init_ls,
                                    smoothing=smoothing, lambda1=2, lambda2=1)
    return ls



def acwe2d(img, coord, iterations, smoothing):
    print('Running: snake_2d (MorphACWE)...')

    range_img = img[:, :, coord[0]]
    range_init_ls = ms.circle_level_set(
        range_img.shape, (coord[2], coord[1]), 5)
    range_ls = ms.morphological_chan_vese(range_img, iterations=iterations,
                                            init_level_set=range_init_ls,
                                            smoothing=smoothing, lambda1=2, lambda2=1)
    # save_img(range_img, range_ls, "acwe_2d_y_slice")


    slices = []
    i = 0
    for row in range_ls:
        for x in row:
            if x == 1:
                slices.append([i, middle_of_line(row)])
                break
        i += 1

    middle = int((slices[-1][0] + slices[0][0]) / 2)
    result = np.zeros(img.shape, dtype=np.uint8)

    for line in slices:
        image_part = img[line[0]]
        init_ls = ms.circle_level_set(
            image_part.shape, (line[1], coord[0]), 5)
        ls = ms.morphological_chan_vese(image_part, iterations=iterations,
                                        init_level_set=init_ls,
                                        smoothing=smoothing, lambda1=2, lambda2=1)
        result[line[0]] = ls
        # if i == middle:
        #     save_img(image_part, ls, "acwe_2d_slice")

    return result


def acwe2d_prev(img, coord, iterations, smoothing):
    print('Running: snake_2d_prev (MorphACWE)...')

    range_img = img[:, :, coord[0]]
    range_init_ls = ms.circle_level_set(
        range_img.shape, (coord[2], coord[1]), 5)
    range_ls = ms.morphological_chan_vese(range_img, iterations=iterations,
                                          init_level_set=range_init_ls,
                                          smoothing=smoothing, lambda1=2, lambda2=1)
    # save_img(range_img, range_ls, "acwe_2d_prev_y_slice")

    slices = []
    i = 0
    for row in range_ls:
        for x in row:
            if x == 1:
                slices.append([i, middle_of_line(row)])
                break
        i += 1

    middle = int((slices[-1][0] + slices[0][0]) / 2)
    middle_index = int(len(slices) / 2)

    middle_img = img[slices[middle_index][0]]
    init_ls = ms.circle_level_set(
        middle_img.shape, (slices[middle_index][1], coord[0]), 5)
    middle_ls = ms.morphological_chan_vese(middle_img, iterations=iterations,
                                         init_level_set=init_ls,
                                         smoothing=smoothing, lambda1=2, lambda2=1)

    result = np.zeros(img.shape, dtype=np.uint8)
    result[slices[middle_index][0]] = middle_ls
    # save_img(middle_img, middle_ls, "acwe_2d_prev_slice")

    prev_ls = middle_ls
    for i in range(middle_index + 1, len(slices)):
        line = slices[i]
        image_part = img[line[0]]
        ls = ms.morphological_chan_vese(image_part, iterations=(iterations // 4),
                                        init_level_set=prev_ls,
                                        smoothing=smoothing, lambda1=2, lambda2=1)
        prev_ls = ls
        result[line[0]] = ls

    prev_ls = middle_ls
    for i in range(middle_index - 1, 0, -1):
        line = slices[i]
        image_part = img[line[0]]
        ls = ms.morphological_chan_vese(image_part, iterations=(iterations // 4),
                                        init_level_set=prev_ls,
                                        smoothing=smoothing, lambda1=2, lambda2=1)
        prev_ls = ls
        result[line[0]] = ls

    return result


def gac3d(img, coord, iterations, smoothing, balloon, threshold):
    print('Running: snake_3d (MorphGAC)...')

    init_ls = ms.circle_level_set(img.shape, (coord[2], coord[1], coord[0]), 5)

    gimage = inverse_gaussian_gradient(img)
    ls = ms.morphological_geodesic_active_contour(gimage, iterations=iterations,
                                                    init_level_set=init_ls,
                                                    smoothing=smoothing, balloon=balloon, threshold=threshold)
    return ls

def gac2d(img, coord, iterations, smoothing, balloon, threshold):
    print('Running: snake_2d (MorphGAC)...')

    range_img = img[:, :, coord[0]]
    range_init_ls = ms.circle_level_set(
        range_img.shape, (coord[2], coord[1]), 5)
    range_gimage = inverse_gaussian_gradient(range_img)
    range_ls = ms.morphological_geodesic_active_contour(range_gimage, iterations=iterations,
                                                            init_level_set=range_init_ls,
                                                            smoothing=smoothing, balloon=balloon, threshold=threshold)
    # save_img(range_img, range_ls, "gac_2d_y_slice")

    slices = []
    i = 0
    for row in range_ls:
        for x in row:
            if x == 1:
                slices.append([i, middle_of_line(row)])
                break
        i += 1

    middle = int((slices[-1][0] + slices[0][0]) / 2)
    result = np.zeros(img.shape, dtype=np.uint8)
    for line in slices:
        image_part = img[line[0]]
        init_ls = ms.circle_level_set(
            image_part.shape, (line[1], coord[0]), 5)

        gimage = inverse_gaussian_gradient(image_part)
        ls = ms.morphological_geodesic_active_contour(gimage, iterations=iterations,
                                                        init_level_set=init_ls,
                                                        smoothing=smoothing, balloon=balloon, threshold=threshold)
        result[line[0]] = ls
        # if i == middle:
        #     save_img(image_part, ls, "gac_2d_slice")


    return result
    


if __name__ == '__main__':

    mode = int(sys.argv[1])
    iterations = int(sys.argv[2])
    smoothing = int(sys.argv[3])
    threshold = float(sys.argv[4])
    balloon = int(sys.argv[5])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img = np.load(dir_path + '/image.npy')
    coord = np.load(dir_path + '/coord.npy')

    ls = []
    start = time.time()
    if mode == 0:
        ls = acwe3d(img, coord, iterations, smoothing)
    elif mode == 1:
        ls = acwe2d(img, coord, iterations, smoothing)
    elif mode == 2:
        ls = gac3d(img, coord, iterations, smoothing, balloon, threshold)
    elif mode == 3:
        ls = gac2d(img, coord, iterations, smoothing, balloon, threshold)
    elif mode == 4:
        ls = acwe2d_prev(img, coord, iterations, smoothing)
    
    end = time.time()
    print("Time: " + str(end - start) + " sec.")

    np.save(dir_path + '/out.npy', ls)
    print("Done.")
