"""
Script to find fundamental matrix from pairs of corresponding points.
- Add function to retrieve key points manually and automatically between two images
- Add function to return fundamental matrix between same images
- Show epipoles, vanishing points and horizon on both images
Thomas Bayley
12/02/21
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys


class Fundamental:

    def __init__(self, img1, img2, method='auto'):
        self.img1 = img1
        self.img2 = img2
        if method == 'auto':
            # find key points using automatic method
            self.kpts1, self.kpts2 = self.auto_corr()
            print('found points by auto correspondence method')
        elif method == 'manual':
            # find keypoints using manual method
            self.kpts1, self.kpts2 = self.auto_corr()
            print('found points by manual correspondence method')
        else:
            print('Error: no such method "{}"'.format(method))

    def auto_corr(self):
        print("Finding auto keypoints")

        MIN_MATCH_COUNT = 10
        INT_TRIGGER = 1  # if this is 1, keypoints found are rounded to integer precision, to make them comparible to manual keypoints

        img1 = self.img1
        img2 = self.img2
        print("Loaded images")

        # Initiate SIFT detector
        # sift = cv2.SIFT()
        sift = cv2.SIFT_create()
        print("Initiated SIFT")

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        print("Found Keypoints")

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            if INT_TRIGGER == 1:
                src_pts = src_pts.astype(int)
                dst_pts = dst_pts.astype(int)
        else:
            print("Not enough matches are found - %d/%d".format(len(good), MIN_MATCH_COUNT))
            src_pts = None
            dst_pts = None

        return src_pts, dst_pts

    def manual_corr(self):
        kpts1 = 0
        kpts2 = 0
        kpts1 = cv2.KeyPoint_convert(kpts1)
        kpts2 = cv2.KeyPoint_convert(kpts2)
        return kpts1, kpts2

    def getFundamental(self, algo=cv2.FM_RANSAC):

        self.F, self.mask = cv2.findFundamentalMat(self.kpts1,
                                                   self.kpts2,
                                                   algo)
        # We select only inlier points
        self.pts1 = np.delete(self.kpts1, np.where(self.mask == 0), axis=0)  # remove outliers and return keypoints
        self.pts2 = np.delete(self.kpts2, np.where(self.mask == 0), axis=0)
        return self.F

    def epilines(self, index):
        # index refers to the image on which epilines need to be drawn (1 or 2)
        try:
            if index == 1:
                lines = cv2.computeCorrespondEpilines(self.pts2.reshape(-1, 1, 2), 2, self.F)
                lines = lines.reshape(-1, 3)
            elif index == 2:
                lines = cv2.computeCorrespondEpilines(self.pts1.reshape(-1, 1, 2), 1, self.F)
                lines = lines.reshape(-1, 3)
            else:
                print('index out of bounds - must be in range (1,2)')
                return
            self.drawLines(lines, index=index)
        except AttributeError:
            print('Error: no such parameter F (make sure to run getFundamental)')

        return

    def drawLines(self, lines, index):
        if index == 1:
            pts = self.pts1
        else:
            pts = self.pts2
        # draw lines
        r, c = self.img1.shape[0:2]
        i = 0
        for r, pt in zip(lines, pts):
            h = i * 1.0 / pts.shape[0]
            colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, 1.0, 1.0))
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            if index == 1:
                cv2.line(self.img1, (x0, y0), (x1, y1), colour, 2)
                cv2.circle(self.img1, tuple(np.squeeze(pt)), 4, colour, -1)
            else:
                cv2.line(self.img2, (x0, y0), (x1, y1), colour, 2)
                cv2.circle(self.img2, tuple(np.squeeze(pt)), 4, colour, -1)
            i += 1
        return
