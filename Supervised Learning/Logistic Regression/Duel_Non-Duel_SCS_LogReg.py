import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_csv('SCS.csv')
df_binary = df[['ACS', 'Rnd', 'Role']]
df_binary.columns = ['ACS', 'Rnd','Role']
df_binary.head()
x = np.array(df[['ACS', 'Rnd']])
X = np.array(x[:,0]).reshape(-1,1)
y = np.array(df_binary['Role'])
print(X.shape)
print(y.shape)
z = np.array(df_binary['ACS'])
logr = LogisticRegression()
logr.fit(X,y)
y_pred = logr.predict(X)
score = logr.score(X,y)
print(rsq)
b = logr.intercept_
w = logr.coef_
y_plot = X*w+b
fig,ax = plt.subplots(1,1,figsize = (18,9))
for i in range(len(y)):
    if (y_pred[i] == 'Duelist'):
        plt.scatter(x[i,1],x[i,0],c = 'b')
        
    else:
        plt.scatter(x[i,1],x[i,0], c = 'r')
         
ax.scatter(100,150,c = 'b',label = 'Duelist')
ax.scatter(100,150,c = 'r', label = 'Non-Duelist')
fig.set_facecolor('#f3edd3')
ax.patch.set_facecolor('#f3edd3')
ax.grid(ls='dotted',lw=.8,color='lightgrey',axis='y',zorder=1)
ax.annotate(xy=(400,50),text=f'Threshold value of ACS: 250',fontname='Andale Mono',fontsize=20)
plt.xlabel('No. of Rounds Played',fontsize=18,fontname='Druk')
plt.ylabel('Average Combat round',fontsize=18,fontname='Druk')
plt.title('Clustering Duelist and Non-Duelist using Logistic Regression',fontsize=24,)
ax.legend()
plt.draw()
plt.savefig('LogReg.png',dpi=300,bbox_inches = 'tight',facecolor='#f3edd3')
plt.show()