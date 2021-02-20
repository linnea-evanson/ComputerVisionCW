"""
Script to find Intrinsic matrix K and Extrinsic matrix [R|t] from sequence of input images.

    [[fx   γ   cx]
K =  [0    fy  cy]
     [0    0    1]]

Thomas Bayley
11/02/21
"""
import numpy as np
import cv2


class Calibration:

    def __init__(self, img_size, patternSize=(4, 4)):
        # tuple of number of inner corners for (row, column) of chessboard
        self.patternSize = patternSize
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        # Prepare obj points, like (0, 0, 0), (1, 0, 0), (2, 0, 0)....., (3, 3, 0)
        self.objp = np.mgrid[0:patternSize[0], 0:patternSize[1], 0:1].T.reshape(-1, 3)  # x,y coordinates
        self.objp = self.objp.astype('float32')
        self.img_size = img_size[0:2]

    def getCorners(self, image):

        # find inner corners of chessboard
        retval, corners = cv2.findChessboardCorners(image, self.patternSize)

        if not retval:
            print('no corners found')
            return

        # cv2.cornerSubPix only takes greyscale image
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # set the needed parameters to find the refined corners
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
        # find more accurate coordinates for corners
        corners = cv2.cornerSubPix(grey, corners, winSize, zeroZone, criteria)
        corners = np.squeeze(corners)
        # If corners are found, add object points and image points
        self.imgpoints.append(corners)
        self.objpoints.append(self.objp)

        # draw corners on chessboard
        for corner in corners:
            corner = tuple(np.squeeze(corner))
            cv2.circle(image, corner, 4, (0, 255, 0), -1)
            cv2.putText(image, 'press any key to continue', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.drawChessboardCorners(image, self.patternSize, corners, retval)

        cv2.imshow('corners', image)
        # press any key to destroy window and continue
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return

    def parameters(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        if ret:
            print('camera matrix', mtx)
            print('distortion coefficients', dist)
            print('rotation vector', rvecs)
            print('translation vector', tvecs)
        else:
            print('unable to calibrate')
        self.mtx = mtx
        self.dist = dist
        return mtx, rvecs, tvecs

    def undist(self, image):
        try:
            undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
            return undistorted
        except AttributeError:
            print('mtx and dist not defined, make sure to get parameters')
        print('Unable to distort image so returning original image')
        return image
