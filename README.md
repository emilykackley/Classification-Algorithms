# Classification-Algorithms

## Linear Discriminant Analysis
- This code uses a model from sklearn to make a prediction using Linear Discriminant Analysis 
  ### Approach
  - Use the wine dataset from sklearn and split it into data and labels
  - Split the data into training and test data (80% training, 20% testing)
  - Scale the data using StandardScalar from sklearn and use to train and test the LDA model
  - Plot data using matplotlib
  ### Limitations
  - LDA is quite sensitive to outliers
  
## Support Vector Machine Classification
- This code uses a model from sklearn to make a prediction using Support Vector Machine Classification 
  ### Approach  
  - Use the iris dataset from sklearn and split it into data and labels
  - Split the data into training and test data (80% training, 20% testing)
  - Create the SVC model with a linear kernel and train the model using the training data
  - Make predictions using the SVC model and test data and determine accuracy
  - Make another SVC model with rbf kernel and repeat above
  ### Comments
  - The linear kernel is much faster, but rbf has better performace
  - The linear kernel is degenerate of rbf so linear will never be more accurage than rbf
  - The accuracy of SVM could be improved by scaling the data before creating the SVM models
  ### Limitations
  - SVM models are sensitive to over-fitting the model
  - SVM requires a lot of memory and computing resources
  
## Natural Language Processing (NLP)
- This code uses an input text file for data input
  ### Approach
  - Create a lemmatizer using the WordNetLemmatizer from NLTK and a tokenizer using the TweetTokenizer from NLTK
  - Separate the contents from input data into tokens and then lemmatize the data
  - Break the tokenized contents into bigrams and calulate the frequency of each bigram int the contents
  - Identify the top 5 bigrams
  - Separate the original contents from the input data into tokenized sentences instead of words like before and then traverse through each sentence to see if one or the top 5 bigrams is in that sentence
  ### Limitations
  - Lemmatization relies on a knowledge base (in this case, WordNet) to obtain the correct base of words. The accuracy depends on the words and the different knoweldge bases. 
  - Lemmatization cannot handle unknown words. 
  - Lemmatization is more precise than stemming, but at the expense of recall
  
## KMeans
- This code uses a model from sklearn to make a prediction using KNN 
  ### Approach
  - Import the iris dataset from sklearn and separate intno data and target values and normalize the data
  - Implement Kmeans for k = 3 and k = 50 by running through a for loop twice, each time changing the value of k
  - Create the kmeans model and then fit with the data
  - Define the labels and centroids/clusters
  ### Comments
  - As the number of clusters increases, the average variance decreases
  - Once the numbers of clusters surpasses the number of features, the algorithm becomes less useful. In this case since the iris dataset has 3 separate classes, k = 3 is ideal
  ### Limitations
  - If you run kmeans on uniform data, it will still result in clusters
  - Kmeans is sensitive to scale
