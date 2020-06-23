#Import packages
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
#Load iris dataset from sklearn
irisdatasets = datasets.load_iris()
#Assign data to x
x = np.array(irisdatasets.data)
#Assign targets to y
y = np.array(irisdatasets.target)

#Split the data into 20% testing data and 80% training data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Create an instance of SVC with a linear kernel
model_linear = SVC(kernel='linear', C=1, decision_function_shape='ovr')
#Train the model with training data (linear kernel)
model_linear.fit(x_train,y_train)
#Make predictions based on testing data (linear kernel)
pred_linear = model_linear.predict(x_test)
#Determine the accuracy of the linear kernel SVC model by comparing the predicted values to the testing y_test values
accuracy_linear = accuracy_score(pred_linear,y_test)

#Create an instance of SVC with rbf kernel
model_rbf = SVC(kernel='rbf')
#Train the model with training data (rbf kernel)
model_rbf.fit(x_train,y_train)
#Make predictions based on testing data (rbf kernel)
pred_rbf = model_rbf.predict(x_test)
#Determine the accuracy of the rbf kernel SVC model by comparing the predicted values to the testing y_test values
accuracy_rbf = accuracy_score(pred_rbf,y_test)

#Print accuracy of SVC with linear kernel
print("The accuracy of SVC with linear kernel is:",accuracy_linear)
#print the accuracy of SVC with rbf kernel
print("The accuracy of SVC with rbf kernel is:",accuracy_rbf)

#The linear kernel is much faster, but rbf has better performance. In this case, there is
#not much data, the performance will not be an issue
#The linear kernel is degenerate of rbf so linear will never be more accurate than rbf
#Both of these are performing relatively the same accuracy so since linear is
#faster, the linear kernel will be the best option.