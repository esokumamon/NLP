# %%


import nltk
# nltk.download('stopwords')
import pandas as pd
import string
from sklearn.naive_bayes import GaussianNB 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB 
import math
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

stopwords=stopwords.words('english')

# load csv file into a data frame
df=pd.read_csv("Youtube05-Shakira.csv")
print()

#display the shape of the dataframe
print(df.shape)
print()

# Display the column names
print(df.info())
print()



# Display the spam count 
print("spam: "+ str(df["CLASS"].value_counts()[1]))
print("ham: "+ str(df["CLASS"].value_counts()[0]))
print()


X=df["CONTENT"].to_list()
Y=df["CLASS"].to_list()


# strip punctuations from strings
for i in range(len(X)):
    X[i] = X[i].translate(str.maketrans('','',"!\"#%&'()*+,-/:;<=>?@[\]^_`{|}~"))
    


#sen1="New way to make money easily and spending 20 minutes daily --&gt; <a href=\"https://www.paidverts.com/ref/Marius1533\">https://www.paidverts.com/ref/Marius1533</a>ï»¿"
#sen2="Lamest World Cup song ever! This time FOR Africa? You mean IN Africa. It wasn&#39;t a Live Aid event or something. She made it seem like a charity case for them instead of a proud moment. WhereÂ was Ricky Martin when you needed him! SMHï»¿"
#v=[sen1, sen2]

# strip punctuations from strings
#for i in range(len(v)):
#    v[i] = v[i].translate(str.maketrans('','',"!\"#%&'()*+,-/:;<=>?@[\]^_`{|}~"))
    

countVec = CountVectorizer(ngram_range=(1,1), stop_words=stopwords)
trainTc = countVec.fit_transform(X)

print(trainTc.toarray())

print()



### 5. Downscale the transformed data using tf-idf and again present highlights of the output (final features) such as the new shape of the data and any other useful information before proceeding.
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(trainTc)
print('After tfidf transform')
print(train_tfidf)

print('Place tfidf into dataframe')
#tfidf_df=pd.DataFrame(train_tfidf.toarray(),columns=countVec.get_feature_names_out())
tfidf_df=pd.DataFrame(train_tfidf.toarray(),columns=countVec.get_feature_names())
print(tfidf_df.head(5))
print(tfidf_df.shape)

### 6. Use pandas.sample to shuffle the dataset, set frac =1
# append class column before shuffle
print('Append class column to dataframe')
tfidf_df['class'] = Y 
print(tfidf_df.head(5))
# tfidf_df.to_excel('tfidf_df.xlsx')
df_shuffle = tfidf_df.sample(frac=1,random_state=200)
print('After shuffle')
print(df_shuffle.head(5))
# df_shuffle.to_excel('df_shuffle.xlsx')
print(df_shuffle.shape)

### 7. Using pandas split your dataset into 75% for training and 25% for testing, make sure to separate the class from the feature(s)
split_index = math.ceil(df_shuffle.shape[0]*0.75)
print('Split Index: '+str(split_index))
# training set feature and class
df_training_x, df_training_y = df_shuffle.iloc[:split_index,:-1], df_shuffle.iloc[:split_index,-1]
print('Training Dataset')
print(df_training_x.shape)
print(df_training_y.shape)
print(df_training_x.head(5))
print(df_training_y.head(5))

# testing set feature and class
df_testing_x, df_testing_y = df_shuffle.iloc[split_index:,:-1], df_shuffle.iloc[split_index:,-1]
print('Testing Dataset')
print(df_testing_x.shape)
print(df_testing_y.shape)
print(df_testing_x.head(5))
print(df_testing_y.head(5))

### 8. Fit the training data into a Naive Bayes classifier. 
# Create Naive Bayes classifier 
classifier = GaussianNB()

# Train the classifier
classifier.fit(df_training_x, df_training_y)


'''
9. Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.
'''
num_folds = 5
accuracy_values = cross_val_score(classifier, df_training_x, df_training_y, scoring='accuracy', cv=num_folds)
print("\nMean results of model accuracy:\n",accuracy_values.mean())

'''
10. Test the model on the test data, print the confusion matrix and the accuracy of the model.
'''
# Train the model on the test data
classifier.fit(df_testing_x, df_testing_y)
# Predict the values for testing data
y_pred = classifier.predict(df_testing_x)
# Define sample labels 
true_labels = df_testing_y
pred_labels = y_pred
# Create confusion matrix 
confusion_mat = confusion_matrix(true_labels, pred_labels)
print('\nConfusion matrix:\n',confusion_mat)
# Compute accuracy 
accuracy = accuracy_score(true_labels, pred_labels)
print("\nAccuracy of the model:\n", accuracy)

'''
11. As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam) and pass them to the classifier and check the results. You can be very creative and even do more happy.
'''
# Define comment as test data 
input_data = [
    'I love you Shakira!!!!!!!', 
    'Good song <3',
    'I am still listenning to it in 2023',
    'Love it',
    'Professor Merlin James is the best professor!',
    'I love Introduction to AI!'
]
# Transform input data using count vectorizer
input_tc = countVec.transform(input_data)
type(input_tc)
print('\ninput_tc:\n',input_tc)
# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
type(input_tfidf)
print('\ninput_tfidf:\n',input_tfidf)
# Predict the output categories
#input_tfidf = pd.DataFrame(data=input_tfidf)
predictions = classifier.predict(input_tfidf.toarray()) 
# Define sample labels 
true_labels = input_data
pred_labels = predictions
# Print the outputs
for input_data, prediction in zip(input_data, predictions):
    print('\nComment : ', input_data, '\nPredicted Class: ', prediction)


# %%