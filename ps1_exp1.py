
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model

boston=load_boston()
(x_train, x_test, y_train, y_test) = train_test_split(np.array(boston.data),boston.target, test_size=0.25, random_state=42)
'''
model00001=Ridge(alpha=0.00001)
model00001.fit(x_train,y_train)
pred00001=model00001.predict(x_test)

model001=Ridge(alpha=0.001)
model001.fit(x_train,y_train)
pred001=model001.predict(x_test)

model01=Ridge(alpha=0.1)
model01.fit(x_train,y_train)
pred01=model01.predict(x_test)

model1=Ridge(alpha=1)
model1.fit(x_train,y_train)
pred1=model1.predict(x_test)



#plt.plot(y_test, pred001,linestyle='none',marker='o',markersize=6,color='green')
#plt.plot(y_test,pred01,linestyle='none',marker='*',markersize=6,color='red')
#plt.plot(y_test,pred1,linestyle='none',marker='+',markersize=6,color='gray') 
plt.plot(y_test,pred00001,linestyle='none',marker='.',markersize=6,color='black') 
#plt.plot(pred001.coef_,alpha=0.001,linestyle='none',marker='o',markersize=6,color='green',label=r'Ridge; $\alpha = 0.001$') # alpha here is for transparency
plt.xlabel('True value')
plt.ylabel('Predicted value')
#plt.title('Alpha=0.001')
#plt.title('Alpha=0.1')
#plt.title('Alpha=1')
plt.title('Alpha=0.00001')
x = np.linspace(0, 50, 40)
y = x
plt.plot(x, y)
plt.show()
in_er00001=np.mean( (pred00001 - y_test)**2 )
lin_er001=np.mean( (pred001 - y_test)**2 )
lin_er01=np.mean( (pred01 - y_test)**2 )
lin_er1=np.mean( (pred1 - y_test)**2 )

'''

poly = preprocessing.PolynomialFeatures(2,interaction_only=True)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)
model_poly00001 = linear_model.Ridge(alpha=0.0001)
model_poly001 = linear_model.Ridge(alpha=0.001)
model_poly01 = linear_model.Ridge(alpha=0.1)
model_poly1 = linear_model.Ridge(alpha=1.0)
# Fit the model
model_poly00001.fit(x_train_poly, y_train)
model_poly001.fit(x_train_poly, y_train)
model_poly01.fit(x_train_poly, y_train)
model_poly1.fit(x_train_poly, y_train)

yp_pred00001 = model_poly00001.predict(x_test_poly)
yp_pred001 = model_poly001.predict(x_test_poly)
yp_pred01 = model_poly01.predict(x_test_poly)
yp_pred1 = model_poly1.predict(x_test_poly)

#plt.plot(y_test,yp_pred00001,linestyle='none',marker='.',markersize=6,color='black') 
#plt.plot(y_test, yp_pred001,linestyle='none',marker='o',markersize=6,color='green')
#plt.plot(y_test,yp_pred01,linestyle='none',marker='*',markersize=6,color='red')
plt.plot(y_test, yp_pred1,linestyle='none',marker='+',markersize=6,color='gray') 

#plt.plot(pred001.coef_,alpha=0.001,linestyle='none',marker='o',markersize=6,color='green',label=r'Ridge; $\alpha = 0.001$') # alpha here is for transparency
plt.xlabel('True value')
plt.ylabel('Predicted value')
#plt.title('Alpha for polynomial=0.001')
#plt.title('Alpha for polynomial=0.1')
plt.title('Alpha for polynomial=1')
#plt.title('Alpha for polynomial=0.00001')
x = np.linspace(0, 50, 40)
y = x
plt.plot(x, y)
plt.show()



# Get the mean squared error between the predicted and actual data
poly_er00001=np.mean( (yp_pred00001 - y_test)**2 )
poly_er001=np.mean( (yp_pred001 - y_test)**2 )
poly_er01=np.mean( (yp_pred01 - y_test)**2 )
poly_er1=np.mean( (yp_pred1 - y_test)**2 )

