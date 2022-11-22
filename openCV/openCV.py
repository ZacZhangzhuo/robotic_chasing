import cv2
import numpy as np
import json
from compas import data

cap = cv2.VideoCapture(1)


lower_blue = np.array([0, 0, 255])
upper_blue = np.array([140, 0, 255])

while True:
    _, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # with open("cvLocations.json", "w", encoding="UTF-8") as j:
    output = []
    for c in contours:
        if len(output) < len(c):
            output = c
    x = 0
    y = 0
    ave_output = []
    if len(output) > 0:
        for o in output:
            x = x + o[0][0]
            y = y + o[0][1]

        ave_output = [x / len(output), y / len(output)]
        data.json_dump(ave_output, ".\\chasing\\cvLocations.json")
    cv2.drawContours(frame, output, -1, (0, 255, 255), 3)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mash", mask)

    key = cv2.waitKey(10)
    if key == 27:
        break
# print(str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "xx" + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()
cv2.destroyAllWindows()
