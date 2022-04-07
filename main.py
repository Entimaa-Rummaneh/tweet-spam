
import pandas as pd
import numpy as np
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.datasets import load_digits

nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
spam_words = ['kamala', 'ocasio', 'Trump', 'democrat', 'election', 'republican', 'romney', 'voting', 'Biden', 'President', 'Obama', 'Beaten', 'Prison', 'drug', 'executions', 'Bias', 'militants', 'bitches']

df = pd.read_csv("train.csv")

# map the 'Quality' value to 0 and the 'Spam' value to 1.
df['Type'] = df.Type.map({'Quality': 0, 'Spam': 1})


def clean_text(text):
    # convert to lower case
    text = text.lower()
    # remove special characters
    text = re.sub("[^a-zA-Z ]", "", text)
    # remove URL
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # remove repeated words
    text = re.sub(r'(.)\1+', r'\1\1', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove stop words
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    # stem words
    text = stem_sentence(text)
    return text


def stem_sentence(sentence):
    ps = PorterStemmer()
    token_words = word_tokenize(sentence)
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(ps.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# checks if the content has URL using regex
def hasURL(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, text)
    # return [x[0] for x in url]
    # if [x[0] for x in url] == [] :
    #    return 0
    # else:
    #    return 1
    return len(url)


def has_hashtag(text):
    regex = r"#"
    hashtag = re.findall(regex, text)
    return len(hashtag)


def has_mentions(text):
    regex = r"@"
    mentions = re.findall(regex, text)
    return len(mentions)


def has_numbers(text):
    regex = r"[0-9]"
    numbers = re.findall(regex, text)
    return len(numbers)


def num_of_characters(text):
    regex = r"\w"
    chars = re.findall(regex, text)
    return len(chars)


def num_of_words(text):
    words = text.split(" ")
    return len(words)


def spam_words_count(text):
    spamword_count = 0
    words = text.split(" ")
    for word in words:
        if word in spam_words:
            spamword_count += 1
    return spamword_count


df['clean_text'] = df['Tweet'].apply(clean_text)
df['URL_count'] = df['Tweet'].apply(hasURL)
df['hashtag_count'] = df['Tweet'].apply(has_hashtag)
df['mentions_count'] = df['Tweet'].apply(has_mentions)
df['numbers_count'] = df['Tweet'].apply(has_numbers)
df['words_count'] = df['Tweet'].apply(num_of_words)
df['characters_count'] = df['Tweet'].apply(num_of_characters)
df['spam_words_count'] = df['Tweet'].apply(spam_words_count)

# writing to csv file
data = pd.DataFrame(df)
data.to_csv('new.csv')

# cosine similarity for clean text
tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
tfidf_matrix = tfidfconverter.fit_transform(df["clean_text"])

cosineSim=[]
for i in range(len(tfidf_matrix.toarray())):
  sim=(cosine_similarity(tfidf_matrix[i], tfidf_matrix)).mean()
  cosineSim.append(sim)

# cosine similarity for location
tfidfconverter2 = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
tfidf_matrix2 = tfidfconverter2.fit_transform(df["clean_text"])

cosineSim_location=[]
for i in range(len(tfidf_matrix2.toarray())):
  sim2=(cosine_similarity(tfidf_matrix2[i], tfidf_matrix2)).mean()
  cosineSim_location.append(sim2)

dict = {'Id': list(df["Id"]),
        'cosineSimilarty': cosineSim,
        # 'clean_text': list(int(df['clean_text'])),
        'following': list(df['following']),
        'followers': list(df["followers"]),
        'actions': list(df["actions"]),
        'is_retweet': list(df["is_retweet"]),
        # 'location': list(int(df["location"])),
        'cosineSim_location': cosineSim_location,
        'URL_count': list(df["URL_count"]),
        'hashtag_count': list(df['hashtag_count']),
        'mentions_count': list(df['mentions_count']),
        'numbers_count': list(df['numbers_count']),
        'words_count': list(df['words_count']),
        'characters_count': list(df['characters_count']),
        'spam_words_count': list(df['spam_words_count']),
        'Type': list(df['Type'])
        }


df1 = pd.DataFrame(dict)

df1.replace([np.inf, -np.inf], np.nan, inplace=True)
df1= df1.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna().astype(int)

x = df1.drop('Type', axis=1).astype(int, errors='ignore')
y = df1['Type'].astype(int, errors='ignore')


def classify(classifier, x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8)
    classifier.fit(X_train, y_train)
    print("The model on train data:")
    y_pred = classifier.predict(X_train)
    Accuracy = accuracy_score(y_train, y_pred) * 100
    Precision = precision_score(y_train, y_pred, average='micro') * 100
    Recall = recall_score(y_train, y_pred) * 100
    F1_score = f1_score(y_train, y_pred) * 100
    print('Accuracy = ', Accuracy)
    print('Precision = ', Precision)
    print('Recall = ', Recall)
    print('F1_score= ', F1_score)
    print(classification_report(y_train, y_pred))
    print("************************************")
    print("The model on test data:")
    y_pred = classifier.predict(X_test)
    Accuracy = accuracy_score(y_test, y_pred) * 100
    Precision = precision_score(y_test, y_pred, average='micro') * 100
    Recall = recall_score(y_test, y_pred) * 100
    F1_score = f1_score(y_test, y_pred) * 100
    print('Accuracy = ', Accuracy)
    print('Precision = ', Precision)
    print('Recall = ', Recall)
    print('F1_score= ', F1_score)
    print(classification_report(y_test, y_pred))
    print("______________________________________________________________")


print("MultinomialNB")
model = MultinomialNB()
classify(model, x, y)

print("RandomForestClassifier")
model = RandomForestClassifier()
classify(model, x, y)

print("GradientBoostingClassifier")
model = GradientBoostingClassifier(n_estimators= 1800,max_features= 'sqrt',max_depth= None)
classify(model, x, y)

print("DecisionTreeClassifier")
model = DecisionTreeClassifier()
classify(model, x, y)
