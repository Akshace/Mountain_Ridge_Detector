#!/usr/local/bin/python3
#
# Authors: [Akshat Arvind (aarvind), Aniket Kale (ankale), Rahul Shamdasani (rshamdas)]
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, April 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return sqrt(filtered_y ** 2)


# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(int(max(y - int(thickness / 2), 0)), int(min(y + int(thickness / 2), image.size[1] - 1))):
            image.putpixel((x, t), color)
    return image


# main program
#
# gt_row = -1
# gt_col = -1
# if len(sys.argv) == 2:
#     input_filename = sys.argv[1]
# elif len(sys.argv) == 4:
#     (input_filename, gt_row, gt_col) = sys.argv[1:]
# else:
#     raise Exception("Program requires either 1 or 3 parameters")
#
# # load in image
# input_image = Image.open(input_filename)
#
# # compute edge strength mask
# # print(argmax(edge_strength(input_image), axis=0))
# edge_strength = edge_strength(input_image)
# imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))


# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
# ridge = [edge_strength.shape[0] / 2] * edge_strength.shape[1]


# BAYES NET
# ##argmax function of numpy gives max values along the axis, if axis = 0; max values in each column is returned
# ridge_max = argmax(edge_strength, axis=0)
def bayes_net(input_image, ridge_max):
    imageio.imwrite("test.jpg", draw_edge(input_image, ridge_max, (255, 0, 0), 8))
    return "Bayes_Net Done"



# VITERBI
def viterbi(col_pixels, row_pixels, trans_p):
    # for i in range(0,col_pixels):
    #     trans_p[i] = 1 - i*100/col_pixels
    total_energy = [0] * col_pixels
    ridge_viterbi = [0] * col_pixels

    for col in range(col_pixels):
        for row in range(row_pixels):
            total_energy[col] = total_energy[col] + edge_strength[row][col]

    image_state = zeros((row_pixels, col_pixels))
    max_state_tracker = zeros((row_pixels, col_pixels))

    ##Calculating emission probability for col 0
    for row in range(row_pixels):
        image_state[row][0] = edge_strength[row][0] / total_energy[0]

    ##Tracking mountain-line using viterbi
    for col in range(1, col_pixels):
        for row in range(row_pixels):
            max_energy = 0
            for pixel in range(-19, 20):
                state = row + pixel
                if 0 <= state < row_pixels:
                    if max_energy < image_state[state][col - 1] * trans_p[abs(pixel)]:  ##Checking current max value with curret state prob
                        max_energy = (image_state[state][col - 1]) * trans_p[abs(pixel)]
                        max_state_tracker[row][col] = state  ##Storing the state(row index) from which the maximize value was achieved
                    image_state[row][col] = (edge_strength[row][col] * max_energy/1000)   ##Updating emission probablities using max probability found
                    ## Divided by 1000 to avoid " overflow encountered in double_scalars " error


# if pixel < 0:
#     ##Multiplying by -1 in pixel to get index in trans_p list
#     if max_energy < state_probab[state][col - 1] * trans_p[-1 * pixel]:
#         max_energy = (state_probab[state][col - 1]) * trans_p[-1 * pixel]
#         max_state[row][col] = state
#     state_probab[row][col] = (edge_strength[row][col] / 1000) * max_energy
# else:
#     if max_energy < state_probab[state][col - 1] * trans_p[pixel]:
#         max_energy = (state_probab[state][col - 1]) * trans_p[pixel]
#         max_state[row][col] = state
#     state_probab[row][col] = (edge_strength[row][col] / 1000) * max_energy




    # Backtracking
    max_l = argmax(image_state, axis=0)
    index_max = max_l[-1]  # Finding max index of last column
    print(index_max)
    for col in range(col_pixels - 1, -1, -1):
        ridge_viterbi[col] = index_max
        index_max = max_state_tracker[int(index_max)][col]  ##Will give the state(pixel) which was stored above

    # output answer
    imageio.imwrite("test.jpg", draw_edge(input_image, ridge_viterbi, (0, 0, 255), 7))
    print("Viterbi Done")
    return image_state, max_state_tracker


# HUMAN PART
def human_viterbi(col_pixels, row_pixels, trans_p, gt_row, gt_col, image_state_human, max_state_human):
    ridge_human = [0] * col_pixels
    # state_prob_human = zeros((row_pixels, col_pixels))
    # max_state_human = zeros((row_pixels, col_pixels))

    ##Maxing human point as 1 in the column so viterbi always grabs this point.
    for row in range(row_pixels):
        image_state_human[row][int(gt_col)] = 0

    image_state_human[int(gt_row)][int(gt_col)] = 1

    for col in range(int(gt_col) + 1, col_pixels):
        for row in range(row_pixels):
            max_energy = 0
            for pixel in range(-49, 50):
                state = row + pixel
                if 0 <= state < row_pixels:
                    if max_energy < image_state_human[state][col - 1] * trans_p[abs(pixel)]:  ##Checking current max value with current state prob
                        max_energy = image_state_human[state][col - 1] * trans_p[abs(pixel)]
                        max_state_human[row][col] = state
                    image_state_human[row][col] = (edge_strength[row][col] * max_energy / 1000)

    for col in range(int(gt_col) - 1, 0, -1):
        for row in range(row_pixels):
            max_energy = 0
            # for pixel in range(-99,100):
            # for pixel in range(-19,20):
            for pixel in range(-49, 50):
                state = row + pixel
                if 0 <= state < row_pixels:
                    if max_energy < image_state_human[state][col + 1] * trans_p[abs(pixel)]:
                        max_energy = image_state_human[state][col + 1] * trans_p[abs(pixel)]
                        max_state_human[row][col] = state
                    image_state_human[row][col] = (edge_strength[row][col] * max_energy / 1000)

    max_l = argmax(image_state_human, axis=0)
    index_max = max_l[-1]  # Finding max index of last column
    for col in range(col_pixels - 1, -1, -1):
        ridge_human[col] = index_max
        index_max = max_state_human[int(index_max)][col]  ##Will give the state(pixel) which was stored above

    # input_image = Image.open(input_filename)
    imageio.imwrite("test.jpg", draw_edge(input_image, ridge_human, (0, 255, 0), 5))
    return "Human_viterbi Done"


gt_row = -1
gt_col = -1
if len(sys.argv) == 2:
    input_filename = sys.argv[1]
elif len(sys.argv) == 4:
    (input_filename, gt_row, gt_col) = sys.argv[1:]
else:
    raise Exception("Program requires either 1 or 3 parameters")

# load in image
input_image = Image.open(input_filename)

# compute edge strength mask
# print(argmax(edge_strength(input_image), axis=0))
edge_strength = edge_strength(input_image)
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# BAYES NET
##argmax function of numpy gives max values along the axis, if axis = 0; max values in each column is returned
ridge_max = argmax(edge_strength, axis=0)
print(bayes_net(input_image, ridge_max))

##VITERBI
image1 = imageio.imread(input_filename)
col_pixels = image1.shape[1]
row_pixels = image1.shape[0]

##Calculating transitional probablities
trans_p = [0] * col_pixels
for i in range(0, col_pixels):
    trans_p[i] = 1 - i * 100 / col_pixels
image_state_human, max_state_human = viterbi(col_pixels, row_pixels, trans_p)

##HUMAN VITERBI
trans_p = [0] * col_pixels
for i in range(0, col_pixels):
    trans_p[i] = 1 - i * 100 / col_pixels
print(human_viterbi(col_pixels, row_pixels, trans_p, gt_row, gt_col, image_state_human, max_state_human))



##1 - 74 77
##2 - 56 152
##3 - 43 160
##4 - 55 141
##5 - 59 93
##6 - 72 95
##7 - 20 83 and 52 24
##8 - 64 125
##9 - 70 169
