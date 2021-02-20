"""
Calculate accuracy from homography or fundamental matrix
Linnea Evanson
09/02/21

"""
import cv2
import numpy as np

class accuracy():
    def acc(self, auto_pts1, auto_pts2, auto_H, img1, img2, good):
        auto_pts1 = np.squeeze(auto_pts1)
        auto_pts1 = np.append(auto_pts1, [[1] for i in range(auto_pts1.shape[0])], axis = 1) #reshape, add coord of 1

        auto_predicted_pts2 = np.zeros((auto_pts1.shape[0], auto_pts1.shape[1], 1))
        for row in range(len(auto_pts1)):
            coord = np.expand_dims(auto_pts1[row], axis = 1)   #add coord of 1 to z dimension, so can multiply by 3x3 h matrix
            auto_predicted_pts2[row] = np.matmul(auto_H , coord) #matrix multiply
            auto_predicted_pts2[row] = auto_predicted_pts2[row] / auto_predicted_pts2[row][2] #divide by z' to get real image coords

        print("Is z coord 1?", auto_predicted_pts2[:4])

        #Accuracy is calculated in terms of precision
        auto_pts2 = np.squeeze(auto_pts2)
        auto_pts2 = np.append(auto_pts2, [[1] for i in range(auto_pts2.shape[0])], axis=1)  # reshape, add coord of 1

        # kpts1 = cv2.KeyPoint_convert(auto_pts1)
        # kpts2 = cv2.KeyPoint_convert(auto_pts2)
        # print("convert worked")
        #
        # #Draw these transformed points:
        # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
        #                    singlePointColor=None,
        #                    matchesMask=None,  # draw only inliers
        #                    flags=2)
        #
        # img3 = cv2.drawMatches(img1, kpts1, img2, kpts2, good, None, **draw_params)
        # print("Saving autocorrespondence image")
        # cv2.imwrite("output_images/auto_correspondences_H_transform.png", img3)


        precision = np.mean( (auto_pts2 - np.squeeze(auto_predicted_pts2)) / auto_pts2)
        rmse =  np.sqrt(np.mean( (auto_pts2 - np.squeeze(auto_predicted_pts2)) **2)  )

        return precision, rmse

    # def visualise(self):
