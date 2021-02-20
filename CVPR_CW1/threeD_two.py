"""
Script for 3D epipolar lines, stereorectification and depth map.
Linnea Evanson
13/02/21

"""
from __future__ import print_function

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from resize import resize

class threeD():
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def stereo_rectify(self):
        img1= cv.cvtColor(self.img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(self.img2, cv.COLOR_BGR2GRAY)

        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)


        # Match keypoints in both images
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Keep good matches: calculate distinctive image features
        # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
        # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        # STEREO RECTIFICATION
        # Calculate the fundamental matrix for the cameras
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

        # We select only inlier points
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]


        # Stereo rectification (uncalibrated variant)
        # Adapted from: https://stackoverflow.com/a/62607343
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        _, H1, H2 = cv.stereoRectifyUncalibrated(
            np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
        )

        # Undistort (rectify) the images and save them
        # Adapted from: https://stackoverflow.com/a/62607343
        img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
        img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
        cv.imwrite("rectified_1.png", img1_rectified)
        cv.imwrite("rectified_2.png", img2_rectified)

        # Draw the rectified images
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(img1_rectified, cmap="gray")
        axes[1].imshow(img2_rectified, cmap="gray")
        axes[0].axhline(1500)
        axes[1].axhline(1500)
        axes[0].axhline(2000)
        axes[1].axhline(2000)
        axes[0].axhline(1750)
        axes[1].axhline(1750)
        axes[0].axhline(1750)
        axes[1].axhline(1750)
        axes[0].axhline(2200)
        axes[1].axhline(2200)

        plt.suptitle("Rectified images")
        plt.savefig("rectified_images.png")
        plt.show()

        return img1_rectified, img2_rectified

    def depth_map_rectified(self, imgL, imgR):

        #Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 11
        min_disp = -128
        max_disp = 128
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 200
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0

        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        disparity_SGBM = stereo.compute(imgL, imgR)

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                      beta=0, norm_type=cv.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)
        cv.imshow("Disparity of Rectified Imgs", resize(disparity_SGBM,30))
        cv.imwrite("disparity_SGBM_norm_GRID.png", disparity_SGBM)

    def depth_map(self):
        imgL = cv.cvtColor(self.img1, cv.COLOR_BGR2GRAY)
        imgR = cv.cvtColor(self.img2, cv.COLOR_BGR2GRAY)

        stereo = cv.StereoBM_create(numDisparities=16,
                                     blockSize=15)  # numDisparities is window size, must be divisible by 16

        disparity = stereo.compute(imgL, imgR)

        local_max = disparity.max()
        local_min = disparity.min()
        print("MAX " + str(local_max))
        print("MIN " + str(local_min))
        disparity_visual = (disparity - local_min) * (1.0 / (local_max - local_min))



        plt.imshow(disparity_visual, 'gray')
        plt.title("Depth map")
        plt.show()