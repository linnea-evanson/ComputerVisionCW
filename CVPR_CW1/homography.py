"""
Script to find homography matrix from pairs of corresponding points.
Linnea Evanson
07/02/21
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

class homography():
    def __init__(self, kpts1, kpts2, img1, img2):
        self.kpts1 = kpts1
        self.kpts2 = kpts2
        self.keypoints1 = cv2.KeyPoint_convert(kpts1)
        self.keypoints2 = cv2.KeyPoint_convert(kpts2)

        self.img1 = img1
        self.img2 = img2

    def get_h(self):
        print("kpts to keypoints convert:",self.kpts1)
        print("dims of kpts:",len(self.kpts1),len(self.kpts2[0]))
        print("kpts type", type(self.kpts1), type(self.kpts1[0]), type(self.kpts1[1]))

        M, mask = cv2.findHomography(self.kpts1, self.kpts2, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Warp source image to destination based on homography
        # im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

        #h, w, _ = self.img1.shape
        #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #dst = cv2.perspectiveTransform(pts, M)

        #self.img2 = cv2.polylines(self.img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           #matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        #Get matches (though we already know they are in order of matching)
        aMatches = []
        for i in range(len(self.keypoints1)):
            aMatches.append([[i], [i]])
        matches = []
        for i in range(len(aMatches)):
            #matches.append(cv2.DMatch(int( aMatches[i][0]), int(aMatches[i][1]) ,i, 0.001 ))
            matches.append(cv2.DMatch( i, i , 0.001 ))
        print("length of matches", len(matches))
        #---Draw
        #img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        img3 = cv2.drawMatches(self.img1, self.keypoints1, self.img2, self.keypoints2, matches, None, **draw_params)

        print("Saving manual correspondences image")
        cv2.imwrite("output_images/manual_correspondences.png", img3)

        return M

    # def display_images(self):
    #     # Display images warped by homography matrix
    #     cv2.imshow("Source Image", im_src)
    #     cv2.imshow("Destination Image", im_dst)
    #     cv2.imshow("Warped Source Image", im_out)
    #     cv2.waitKey(0)
