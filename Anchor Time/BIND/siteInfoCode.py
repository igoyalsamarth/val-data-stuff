import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#newimage = cv2.imread('bind.webp')
#oldimage = cv2.cvtColor(newimage,cv2.COLOR_BGR2RGB)
#image = cv2.resize(oldimage,(1000,1000))
siteB = np.array([[120,150],[520,150],[520,900],[120,900]])
siteA = np.array([[530,150],[900,150],[900,900],[530,900]])
#cv2.drawContours(image,[siteB], -1, (0,255,0), 2)
#cv2.putText(image,text = 'Bomb Site B', org = (250,335), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (0,255,0),thickness = 2,lineType = cv2.LINE_AA)
#cv2.drawContours(image,[siteA], -1, (255,0,0), 2)
#cv2.putText(image,text = 'Bomb Site A', org = (735,360), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (255,0,0),thickness = 2,lineType = cv2.LINE_AA)
#cv2.imshow('Contours',image)
#plt.imshow(image)

df = pd.read_csv('skillsAnchor_Bind.csv')
xloc = np.array(df['locX'])
yloc = np.array(df['locY'])
r= np.array(df['round'])
time = np.array(df['TimeMillis'])
#print(len(r))
x = 0
y = 0
T = []
for i in range(1,13):
    print(i)
    df1 = (df.loc[df['round'] == i])
    print(df1)
    y = len(df1)
    if(y == 1):
        print(time[x])
        x = x+1
        continue
    for j in range(x,x+y):
        #print(j)
        if((cv2.pointPolygonTest(siteB,(xloc[j],yloc[j]),False) != -1) == True):
            print('A site')
        else:
            print('B site')
    x = x+y