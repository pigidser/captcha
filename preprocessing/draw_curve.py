
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("0009.png") #1000x1000 pixel

#lets say these are my black pixels in a white image.
x = np.array([0, 50, 100])
y = np.array([50, 55, 50])

# Initilaize y axis
lspace = np.linspace(0,  img.shape[0]-1, img.shape[0])

#calculate the coefficients.
z = np.polyfit(x, y, 2)

#calculate x axis
line_fitx = z[0]*lspace**2 + z[1]*lspace+ z[2]

# # Create axes figure
# ax1 = plt.axes()

# #Show image
# ax1.imshow(img, aspect='auto')

# #Draw Polynomial over image
# ax1.plot(line_fitx,lspace, color='blue');



verts = np.array(list(zip(line_fitx.astype(int),lspace.astype(int))))
cv2.polylines(img,[verts],False,(0,255,255),thickness=2)

cv2.imshow("Output", img)
cv2.waitKey(0)

# ax2 = plt.axes()
# ax2.imshow(img)
# plt.show()
# cv2.waitKey()