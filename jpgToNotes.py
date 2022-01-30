import cv2
import numpy as np
import pytesseract
import os


def produce():
    image = input("Name of file: ")
    img = cv2.imread(image)
    kernel = np.ones((7, 7), np.uint8)

    # convert to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)

    # lower bound and upper bound for pink highlight color
    lower_bound = np.array([110, 0, 150])
    upper_bound = np.array([200, 200, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Segment only the detected region
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    segmented_name = "segmented_img{}.jpg".format(os.getpid())
    cv2.imwrite(segmented_name, segmented_img)

    text = pytesseract.image_to_string(segmented_name, lang='eng')

    outputFile = "output{}.txt".format(os.getpid())
    with open(outputFile, "w") as f:
        for line in text.splitlines():
            f.write(line + "\n")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
