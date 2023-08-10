
import cv2
import numpy as np


image_path = 'D:/R-C.jpg'  
image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([150, 255, 255])
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = 150
plate_images = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_contour_area:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Detected License Plate', image)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()

