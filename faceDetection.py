import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #this file cantains data of several faces (reinforcement learning) provided by opencv
img = cv2.imread("lena.jpg") #Enter the name of image with it's typew
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #this convorts image into gray scale

faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow("RESULT",img)
cv2.waitKey(0)
