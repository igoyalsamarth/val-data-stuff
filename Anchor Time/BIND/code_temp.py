import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

newimage = cv2.imread('bind.webp')
oldimage = cv2.cvtColor(newimage,cv2.COLOR_BGR2RGB)
image = cv2.resize(oldimage,(1080,1000))

siteB = np.array([[247,220],[400,220],[400,500],[310,500],[280,460],[230,460],[230,395],[210,395],[210,305],[247,305]])
siteA = np.array([[610,300],[690,300],[690,230],[780,230],[780,250],[960,250],[960,500],[610,500]])
cv2.drawContours(image,[siteB], -1, (0,255,0), 2)
cv2.putText(image,text = 'Bomb Site B', org = (250,335), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (0,255,0),thickness = 2,lineType = cv2.LINE_AA)
cv2.drawContours(image,[siteA], -1, (255,0,0), 2)
cv2.putText(image,text = 'Bomb Site A', org = (735,360), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (255,0,0),thickness = 2,lineType = cv2.LINE_AA)
#cv2.imshow('Contours',image)

plt.imshow(image)

df = pd.read_csv('GEvsME_Bind.csv')
xloc = np.array(df['locationX'])
yloc = np.array(df['locationY'])

for i in range(len(xloc)):
    if(cv2.pointPolygonTest(siteA, (xloc[i],yloc[i]), False) != -1 or cv2.pointPolygonTest(siteB, (xloc[i],yloc[i]), False) != -1 ):
        plt.scatter(xloc[i],yloc[i])

plt.show()