import cv2
import numpy as np
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

img = cv2.imread('img.jpg')
img = cv2.resize(img, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(2)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame  = np.flip(frame, axis = 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    u_black = np.array([104, 153, 70])
    l_black = np.array([30, 30, 0])

    mask_1 = cv2.inRange(frame, l_black, u_black)
    mask_2=cv2.bitwise_not(mask_1)

    res1 = cv2.bitwise_and(frame, frame, mask=mask_2)

    res2=cv2.bitwise_and(img, img, mask = mask_1)

    
    final_output = cv2.addWeighted(res1,1 ,res2 ,1 , 0)
    output_file.write(final_output)
    cv2.imshow('MAGIC', final_output)
    cv2.waitKey(1)

cap.release()
output_file.release()
cv2.destroyAllWindows()
