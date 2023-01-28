#https://www.kaggle.com/code/radustoicescu/use-keras-to-classify-sea-lions-0-91-accuracy/notebook
import argparse
import os
import cv2
import numpy as np
import skimage


def get_label(img, mask, threshold=0.96):
    # Perform match operations.
    res = cv2.matchTemplate(img, mask, cv2.TM_CCORR_NORMED)
    w, h, _ = mask.shape
    # Specify a threshold
    # Store the coordinates of matched area in a numpy array
    loc = np.where(res >= threshold)

    # Draw a rectangle around the matched region.
    bb = []
    for pt in zip(*loc[::-1]):
        img = cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), -1)
        bb.append((pt[0] + w / 2, pt[1] + h / 2))

    return img, bb

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='This program allow user to train RTS on a object detection dataset')
    parser.add_argument('-img', '--img_path',
                        help='the path to the data. it must contains a images and a labels folder')
    parser.add_argument('-dot', '--dot_path',
                        help='the path to the data. it must contains a images and a labels folder')
    parser.add_argument('-out', '--results_path',
                        help='the path where results will be saved')

    args = parser.parse_args()

    img_list = sorted(os.listdir(args.img_path))
    img_list_dot = sorted(os.listdir(args.dot_path))

    for file in img_list:

        if not os.path.exists(os.path.join(args.dot_path, file)):
            continue

        print(file)
        img_dot_color = cv2.imread(os.path.join(args.dot_path, file))
        img_dot = cv2.cvtColor(img_dot_color, cv2.COLOR_BGR2GRAY)

        img = cv2.imread(os.path.join(args.img_path, file))

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = img_dot.copy()
        mask[mask < 20] = 0
        mask[mask > 0] = 255

        img_gray[mask == 0] = 0
        img_dot[mask == 0] = 0

        only_dot = cv2.absdiff(img_gray, img_dot)

        kernel = np.ones((5, 5), np.uint8)

        # opening the image
        opening = cv2.morphologyEx(only_dot, cv2.MORPH_OPEN, kernel, iterations=1)
        blobs = skimage.feature.blob_log(opening, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

        new_file = open(os.path.join(args.results_path,  file.split(".")[0] + ".txt"), "w")
        for blob in blobs:
            y, x, s = blob
            g, b, r = img_dot_color[int(y)][int(x)][:]

            y -= 16
            x -= 16
            w = 32
            h = 32

            label = -1
            if r > 200 and g < 50 and b < 50:  # RED
                label = 0
            elif r > 200 and g > 200 and b < 50:  # MAGENTA
                label = 1
            elif r < 100 and g < 100 and 150 < b < 200:  # GREEN
                label = 2
            elif r < 100 and 100 < g and b < 100:  # BLUE
                label = 3
            elif r < 150 and g < 50 and b < 100:  # BROWN
                label = 4

            if label == -1:
                continue

            new_file.writelines("{0} {1} {2} {3} {4}\n".format(label, x, y, w, h))
        new_file.close()



