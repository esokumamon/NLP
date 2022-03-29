


import nltk
import pandas as pd
import string
from sklearn.naive_bayes import GaussianNB 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords
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




tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(trainTc)

print(train_tfidf.shape)

tfidf_df=pd.DataFrame(train_tfidf.toarray(),columns=countVec.get_feature_names())
print(tfidf_df.head(5))

