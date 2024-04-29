import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

newimage = cv2.imread('bind.webp')
oldimage = cv2.cvtColor(newimage,cv2.COLOR_BGR2RGB)
image = cv2.resize(oldimage,(1000,1000))

siteB = np.array([[120,150],[520,150],[520,900],[120,900]])
siteA = np.array([[530,150],[900,150],[900,900],[530,900]])
cv2.drawContours(image,[siteB], -1, (0,255,0), 2)
cv2.putText(image,text = 'Bomb Site B', org = (250,335), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (0,255,0),thickness = 2,lineType = cv2.LINE_AA)
cv2.drawContours(image,[siteA], -1, (255,0,0), 2)
cv2.putText(image,text = 'Bomb Site A', org = (735,360), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (255,0,0),thickness = 2,lineType = cv2.LINE_AA)
#cv2.imshow('Contours',image)

#plt.imshow(image)

df = pd.read_csv('VLTrite2aceAnchor_BindvsGE.csv')
xloc = np.array(df['locationX'])
yloc = np.array(df['locationY'])
r= np.array(df['roundNumber'])
time = np.array(df['roundTimeMillis'])
#print(len(r))
x = 0
y = 0
z = 0
T = []
for i in range(1,13): #change 13 to (whatever the final number of rounds are in second half - 12)
    #print(i)
    df1 = (df.loc[df['roundNumber'] == i]) # == (i+12) for second half
    #print(df1)
    y = len(df1)
    #print('x',x)
    #print('x+y',x+y-1)
    if(y == 1):
        T.append(time[x]) #append1
        x = x+y
        continue
    if(((cv2.pointPolygonTest(siteB,(xloc[x],yloc[x]),False) != -1) == True) and (cv2.pointPolygonTest(siteB,(xloc[x+1],yloc[x+1]),False) != -1) == False):
        T.append(time[x]) #append2
        x = x+y
        continue
    if(((cv2.pointPolygonTest(siteA,(xloc[x],yloc[x]),False) != -1) == True) and (cv2.pointPolygonTest(siteA,(xloc[x+1],yloc[x+1]),False) != -1) == False):
        T.append(time[x]) #append3
        x = x+y
        continue
    for j in range(x,x+y):
        #print(j)
        if(((cv2.pointPolygonTest(siteB,(xloc[x],yloc[x]),False) != -1) == True) and ((cv2.pointPolygonTest(siteB,(xloc[j],yloc[j]),False) != -1) == False)):
            z = j-1
            break
        elif(((cv2.pointPolygonTest(siteA,(xloc[x],yloc[x]),False) != -1) == True) and ((cv2.pointPolygonTest(siteA,(xloc[j],yloc[j]),False) != -1) == False)):
            z = j-1
            break    
        elif(((cv2.pointPolygonTest(siteA,(xloc[x],yloc[x]),False) != -1) == True) and ((cv2.pointPolygonTest(siteA,(xloc[x+y-1],yloc[x+y-1]),False) != -1) == True)):
            z = x+y-1
            break
        elif(((cv2.pointPolygonTest(siteB,(xloc[x],yloc[x]),False) != -1) == True) and ((cv2.pointPolygonTest(siteB,(xloc[x+y-1],yloc[x+y-1]),False) != -1) == True)):
            z = x+y-1
            break
    #print('z = ',z)
    T.append(time[z])
        
    x = x+y
T = np.array(T)
T = ((T/(1000)))
print(T.shape)
anchor = round(np.average(T),2)
print(anchor)
y = np.arange(1,13,1) #arange(13,25,1) for second half
fig = plt.figure(figsize = (18,9),facecolor = '#fff3d1')
ax = plt.axes()
ax.set_facecolor('#fff3d1')
plt.grid(color = '#a3a3a3', linestyle='--',linewidth=0.2)
plt.ylim([0,120])
plt.plot(y,T,label = 'Round Anchor Time')
plt.hlines(anchor,xmin = 1,xmax = 12,color = 'red',linestyle = ':',label = 'Average Anchor Time') #xmin = 13,xmax = 24
plt.annotate(f'Avg. Anchor Time = {anchor}s', (10.2,105),size = 14)
plt.title('VLT rite2ace || Site Anchor Time || BIND || vs GE',size = 18)
plt.xlabel('Round number ->',size = 16)
plt.ylabel('Time [s] ->',size = 16)

plt.legend(fontsize = '12')
plt.savefig('VLTrite2aceAnchorTimevsGE.png',dpi = 600,bbox_inches = 'tight',facecolor = 'auto')
plt.show()