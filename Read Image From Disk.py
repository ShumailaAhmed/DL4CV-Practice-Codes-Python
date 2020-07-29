#Read and example image from disk and display it on screen
import cv2
image= cv2.imread('example.jpg')
print(image.shape)
cv2.imshow("Img",image)
cv2.waitKey(0)
#accessing individual pixels
b,g,r=image[20,100]
print(b)
print(g)
print(r)
