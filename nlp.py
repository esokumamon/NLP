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
tfidf_df=pd.DataFrame(train_tfidf.toarray(),columns=countVec.get_feature_names_out())
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


# %%
