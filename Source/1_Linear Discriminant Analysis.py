#Import packages
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Import wine dataset from sklearn
winedatasets = datasets.load_wine()
#Separate data and target from wine dataset and define target names
x=winedatasets.data
y=winedatasets.target
target_names= winedatasets.target_names
#Split data into test data and training data. 80% training, 20% testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#Create scaler model
sc = StandardScaler()
#Scale training and testing data
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#Create the LDA model
model = LinearDiscriminantAnalysis(n_components=2)
#Train the model and test with x_test
labels = model.fit(x_train,y_train).transform(x_test)

#Plot the 3 classes with orange, turquoise and purple
colors = ['orange', 'turquoise', 'purple']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(labels[y_test == i, 0], labels[y_test == i, 1], alpha=.8, color=color,
                label=target_name)
#Define legend and title
plt.legend(loc='lower right', shadow=False, scatterpoints=1)
plt.title('Linear Discriminant Analysis of Wine Dataset')

#Show plot
plt.show()