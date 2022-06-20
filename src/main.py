import cv2
import numpy as np


def find_no_coins():

	img = cv2.imread('../data/image.png')

	kernel1 = np.ones((5, 5), np.uint8)

	img_erode = cv2.erode(img, kernel1, iterations=6) 
	result = cv2.dilate(img_erode, kernel1, iterations=4)

	params = cv2.SimpleBlobDetector_Params()

	params.minThreshold = 10
	params.filterByArea = True
	params.minArea = 100
	params.filterByCircularity = True
	params.minCircularity = 0.2
	params.filterByConvexity = True
	params.minConvexity = 0.6
	params.filterByInertia = True
	params.minInertiaRatio = 0.01

	detector = cv2.SimpleBlobDetector_create(params)

	keypoints = detector.detect(cv2.bitwise_not(result.copy()))

	blank = np.zeros((1, 1))
	blobs = cv2.drawKeypoints(result, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	number_of_coins = len(keypoints)
	print("The number of coins in the image are: ", number_of_coins)

	cv2.imshow("Filtering Circular Blobs Only", blobs)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":

	find_no_coins()
