#!/usr/bin/env python3

import argparse
import os
import random
import cv2


import numpy as np

classes = {}


class BoundingBox:
    def __init__(self, points):
        self.points = points

    def get_center(self):
        x = np.mean(self.points[:, 0])
        y = np.mean(self.points[:, 1])
        return int(x), int(y)

    def get_width_height(self):
        width = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        height = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        return int(width), int(height)


def generate_random_image(img, res):

    w = img.shape[1]
    h = img.shape[0]
    x_pad = random.randint(0, w - res)
    y_pad = random.randint(0, h - res)

    new_img = img[y_pad:(y_pad + res), x_pad:(x_pad + res)]
    new_img = cv2.resize(new_img, (224, 224))
    return new_img, x_pad, y_pad


def sub_image_label(labels, x_pad, y_pad, res):
    # filter out all element that are not contained in the sub image
    element_in_sub_image = list(filter(lambda bb: (x_pad <= bb['x'] < x_pad + res and
                                                   y_pad <= bb['y'] < y_pad + res), labels))
    # map the new coordinate of the center of gravity
    return list(map(lambda bb: (bb['label'],
                                bb['x'] - x_pad,
                                bb['y'] - y_pad,
                                bb['w'],
                                bb['h']), element_in_sub_image))

def generate_X_Y(img, labels, nb_sub_img, out_dir_label, out_dir_img, name):

    for i in range(nb_sub_img):

        res = random.randint(224, 400)

        sub_img, x_pad, y_pad = generate_random_image(img, res)
        sub_labels = sub_image_label(labels, x_pad, y_pad, res)

        cv2.imwrite(os.path.join(out_dir_img,name + "_" + str(i) + ".jpg"), sub_img)
        new_file = open(os.path.join(out_dir_label, name + "_" + str(i) + ".txt"), "w")

        counts = np.array([0., 0., 0., 0., 0.])

        for label in sub_labels:
            counts[int(label[0])] += 1

        new_file.writelines("{0} {1} {2} {3} {4}\n".format(int(counts[0]),
                                                           int(counts[1]),
                                                           int(counts[2]),
                                                           int(counts[3]),
                                                           int(counts[4])))
        new_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to select images in Dota dataset with ship in it')
    parser.add_argument('-i', '--img_path', help='the path to the images folder')
    parser.add_argument('-l', '--label_path', help='the path to the labels folder')
    parser.add_argument('-o', '--out_path', help='the path where images will be stored')
    parser.add_argument('-n', '--nb_sub_image', help='the number of sub images that will be created per images')

    args = parser.parse_args()

    stored_image_path = os.path.join(args.out_path, "images")
    stored_label_path = os.path.join(args.out_path, "labels")

    os.makedirs(stored_image_path)
    os.makedirs(stored_label_path)

    img_list = os.listdir(args.img_path)
    label_list = os.listdir(args.label_path)

    for image_filename in img_list:

        filename = image_filename.split('.')[0] + '.txt'

        if filename in label_list:
            with open(os.path.join(args.label_path, filename)) as file:
                lines = file.readlines()
            img = cv2.imread(os.path.join(args.img_path, image_filename))

            img_bb = []
            for line in lines:

                line = line.split(' ')
                img_bb.append({
                    "x": float(line[1]),
                    "y": float(line[2]),
                    "w": float(line[3]),
                    "h": float(line[4]),
                    "label": int(line[0])
                })

            print("parsing file:", filename)
            generate_X_Y(img, img_bb, int(args.nb_sub_image),
                         stored_label_path, stored_image_path, filename.split('.')[0])