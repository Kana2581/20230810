import cv2
import numpy as np


def get_four_corners(contour):
  
    hull = cv2.convexHull(contour)
    
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx_corners = cv2.approxPolyDP(hull, epsilon, True)

  
    if len(approx_corners) != 4:
        raise ValueError("无法获得四个角点")

  
    sorted_corners = sort_corners_clockwise(approx_corners)

    return sorted_corners

def sort_corners_clockwise(corners):
    # 对四个角点进行排序，使其按照顺时针方向排列

    centers = np.mean(corners, axis=0)
    corners = corners - centers
    corners = sorted(corners, key=lambda coord: np.arctan2(coord[0][0], coord[0][1]))
    corners = np.array(corners) + centers

    return corners



def perspective_transform(image, pts):
    width = 300  # 输出图像的宽度
    height = 100  # 输出图像的高度

    dst_pts = np.array([[0, 0], [0, height - 1], [width - 1, height - 1],[width - 1, 0]], dtype="float32")
 
    
    true_pts=np.array([pts[0][0], pts[1][0],pts[2][0],pts[3][0]], dtype="float32")
    M = cv2.getPerspectiveTransform(true_pts, dst_pts)
    print(true_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

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
       # x, y, w, h = cv2.boundingRect(contour)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow('Detected License Plate', image)

        plate_corners = get_four_corners(contour)
        corrected_plate = perspective_transform(image, plate_corners)
        plate_roi = corrected_plate[0:99, 0:299]
        plate_images.append(plate_roi)

scale_factor = 2
enlarged_plate_images = []
for plate_roi in plate_images:
    enlarged_plate = cv2.resize(plate_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    enlarged_plate_images.append(enlarged_plate)

for enlarged_plate in enlarged_plate_images:
    cv2.imshow('Enlarged License Plate', enlarged_plate)
    cv2.waitKey(0)

cv2.destroyAllWindows()

