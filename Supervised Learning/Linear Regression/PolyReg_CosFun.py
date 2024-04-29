import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.arange(0,20,0.5).reshape(-1,1)
y = np.cos(X/2)

poly_reg = PolynomialFeatures(degree = 12)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

score = lin_reg2.score(X_poly,y)
intc = lin_reg2.intercept_
slope = lin_reg2.coef_
print(slope[:,1])
fig,ax = plt.subplots(figsize = (18,9))

plt.scatter(X,y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'red')

fig.set_facecolor('#f3edd3')
ax.patch.set_facecolor('#f3edd3')
ax.grid(ls='dotted',lw=.8,color='lightgrey',axis='y',zorder=1)
ax.annotate(xy=(0,1),text='cos(0) = 1',fontsize=15)
ax.annotate(xy=(18.5, -0.88),text='cos(19.5) = 0.9426 ',fontsize=15)
ax.annotate(xy=(5,0.75),text=f'R-Squared = {round(score,2)}\n',fontname='Andale Mono',fontsize=15)
plt.xlabel('Degree (\u03B8)',fontsize=18,fontname='Druk')
plt.ylabel('Value (f(\u03B8))',fontsize=18,fontname='Druk')
plt.title(f'COS Function \n y = {slope[:,1]}*x + {slope[:,2]}*x\u00b2 + {slope[:,3]}*x\u00b3 + {slope[:,4]}*x\u2074 + {slope[:,5]}*x\u2075 + {slope[:,6]}*x\u2076 + {intc}',fontsize=15,)
#plt.show()
plt.draw()
plt.savefig('COSFunction.png',dpi=300,bbox_inches = 'tight',facecolor='#f3edd3')
plt.show()