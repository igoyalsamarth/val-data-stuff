from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('SCS.csv')
df_binary = df[['ACS', 'ADR']]
df_binary.columns = ['ACS', 'ADR']
df.head()

X = np.array(df_binary['ADR']).reshape(-1,1)
y = np.array(df_binary['ACS'])

regr = LinearRegression()
regr.fit(X,y)

score = regr.score(X,y)
intc = regr.intercept_
slope = regr.coef_
y_pred = regr.predict(X)

fig,ax = plt.subplots(figsize = (18,18))
plt.scatter(X,y)
plt.scatter(161.7,261.3, color = 'black')
plt.scatter(170.5,274.4, color = 'black')
plt.scatter(173.6,273.6, color = 'black')
plt.scatter(172.3,272.3, color = 'black')

plt.plot(X,y_pred,c='red',linestyle='--',dashes=(5,5))

fig.set_facecolor('#f3edd3')
ax.patch.set_facecolor('#f3edd3')
ax.grid(ls='dotted',lw=.8,color='lightgrey',axis='y',zorder=1)
ax.annotate(xy=(151, 261),text='GES Skrossi',fontsize=10)
ax.annotate(xy=(160, 275),text='OGT shooter',fontsize=10)
ax.annotate(xy=(175, 272),text='VLT Deathmaker',fontsize=10)
ax.annotate(xy=(173, 269),text='ENGM Rawfiul',fontsize=10)
ax.annotate(xy=(40,270),text=f'R-Squared = {round(score,2)}\nRegression Equation: y = {intc} + {slope} * x ',fontname='Andale Mono',fontsize=18)
plt.xlabel('Average Damange per Round (ADR)',fontsize=18,fontname='Druk')
plt.ylabel('Average Combact Score (ACS)',fontsize=18,fontname='Druk')
plt.title('Evaluating the Relationship Between ADR  and ACS\n(Data of Skyesports Championship Series)',fontsize=24)
plt.draw()
plt.savefig('SCS_ACSvsADR_LinReg.png',dpi=300,bbox_inches = 'tight',facecolor='#f3edd3')
plt.show()
