import cv2
import numpy as np

# Load the black background image
# image = cv2.imread(r'C:\Users\zac\Desktop\temp\test.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread(r'C:\Users\zac\Desktop\temp\1\10-200.bmp')
image = cv2.imread(r'C:\Users\zac\Desktop\temp\1\10-200.bmp')

# Define the region of interest (ROI) coordinates
x1, y1 = 480, 280  # Top-left coordinate
x2, y2 = 1270, 825  # Bottom-right coordinate

# Crop the image using the ROI coordinates
image = image[y1:y2, x1:x2]

# frame = cv2.resize(frame, (640, 600))
# get current positions of the trackbars


# convert color to hsv because it is easy to track colors in this color model
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_hsv = np.array([0, 0, 0])
higher_hsv = np.array([0, 255, 170])
# Apply the cv2.inrange method to create a mask
mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
# Apply the mask on the image to extract the original color
image = cv2.bitwise_and(image, image, mask=mask)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise (optional)
image_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Apply the Hough Circle Transform
circles = cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=30, minRadius=10, maxRadius=50)

# Convert the circles to integer coordinates
circles = np.uint16(np.around(circles))

# Create a new RGB image to draw the circles on
rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Draw the detected circles
if circles is not None:
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        # Draw the circle outline in red
        cv2.circle(rgb_image, center, radius, (0, 0, 255), 2)

# Display the image with detected circles
cv2.imshow('Detected Circles', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

