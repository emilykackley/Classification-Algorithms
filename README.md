# Classification-Algorithms

## Linear Discriminant Analysis
- This code uses a model from sklearn to make a prediction using Linear Discriminant Analysis 
  ### Approach
  - Use the wine dataset from sklearn and split it into data and labels
    
      *#Import wine dataset from sklearn*
    
      *winedatasets = datasets.load_wine()*
  - Split the data into training and test data (80% training, 20% testing)
      *#Separate data and target from wine dataset and define target names*
      
      *x=winedatasets.data*
      
      *y=winedatasets.target*
      
      *target_names= winedatasets.target_names*
      
      *#Split data into test data and training data. 80% training, 20% testing data*
      
      *x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)*
  - Scale the data using StandardScalar from sklearn and use to train and test the LDA model
  - Plot data using matplotlib
  
