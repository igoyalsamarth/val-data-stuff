import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('LinReg.csv')
df_binary = df[['ACS', 'ADR']]
df_binary.columns = ['ACS', 'ADR']
df_binary.head()
X = np.array(df_binary['ADR']).reshape(-1,1)
y = np.array(df_binary['ACS'])

regr = LinearRegression().fit(X,y)
score = regr.score(X,y)
intc = regr.intercept_
slope = regr.coef_
y_pred = regr.predict(X)

#plotting
fig,ax = plt.subplots(figsize = (18,18))
plt.scatter(X,y)
plt.plot(X,y_pred,c='red',linestyle='--',dashes=(5,5))

#plot decoration
fig.set_facecolor('#f3edd3')
ax.patch.set_facecolor('#f3edd3')
ax.grid(ls='dotted',lw=.8,color='lightgrey',axis='y',zorder=1)
ax.annotate(xy=(175.4,281.3),text='vakk',fontsize=20)
ax.annotate(xy=(70,121),text='Zehradieux ',fontsize=20)
ax.annotate(xy=(80,270),text=f'R-Squared = {round(score,2)}\nThe regression equation: y = {intc} + {slope} * x ',fontname='Andale Mono',fontsize=20)
plt.xlabel('Average Damange per round',fontsize=18,fontname='Druk')
plt.ylabel('Average Combact Score',fontsize=18,fontname='Druk')
plt.title('Evaluating the Relationship Between ADR  and ACS',fontsize=24,)

#plt.show()
plt.draw()
plt.savefig('LinReg.png',dpi=300,bbox_inches = 'tight',facecolor='#f3edd3')
