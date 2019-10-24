import csv
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer
import nltk


from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}


scores = []
original_reviews = []
summaries = []
with open('B007JFMH8M.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        scores.append(row[6])
        summaries.append(row[8])
        original_reviews.append(row[9])


stop_words = set(stopwords.words('english')) 
def text_cleaner(text):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

cleaned_text = []
reviews = [0] * (len(original_reviews))
for i in range(len(original_reviews)):
    reviews[i] = text_cleaner(original_reviews[i])

def summary_cleaner(text):
    newString = re.sub('"','', text)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = newString.lower()
    tokens=newString.split()
    newString=''
    for i in tokens:
        if len(i)>1:                                 
            newString=newString+i+' '  
    return newString

#Call the above function
cleaned_summary = []
for i in range(len(summaries)):
    summaries[i] = summary_cleaner(summaries[i])

#reviews = [x.split(" ") for x in reviews]

# training model
## Use Doc2Vec
reviews= [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews)]
model = Doc2Vec(reviews, vector_size=5, window=2, min_count=1, workers=4)
l = []
for idx in range(len(reviews)):
    l.append(model.docvecs[idx])

# get vector data
X =  np.array(l)
"""
## Use Word2Vec
model = Word2Vec(reviews, min_count=1)
def vectorizer(sent, m):
    vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = m[w]
            else: vec = np.add(vec, m[w])
            numw += 1
        except:
            pass
    return np.asarray(vec)/numw

l = []
for i in reviews:
    l.append(vectorizer(i, model))

# get vector data
X =  np.array(l)
"""

"""
# K means
kmeans = []
wcss = []
# Code that find the num cluster
for NUM_CLUSTERS in range(1,20):
    print("num cluster", NUM_CLUSTERS)
    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,20), wcss)
plt.title("Elbow Method")
plt.xlabel("num cluster")
plt.ylabel("WCSS")
plt.show()
"""
wcss = []
NUM_CLUSTERS = 3
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS, max_iter = 100,init = 'k-means++', random_state = 42)
kmeans.fit(X)
labels = kmeans.fit_predict(X)

# Metrics
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print ("Silhouette_score: ")
print (silhouette_score)

# Compute the closest data
alldistances = kmeans.fit_transform(X)
alldistances = [np.sum(dist**2) for dist in alldistances]
closest_idx = [-1]*NUM_CLUSTERS
for i in range(len(alldistances)):
    cluster = labels[i]
    if closest_idx[cluster] == -1:
        closest_idx[cluster] = i
    else:
        if alldistances[i] < alldistances[closest_idx[cluster]]:
            closest_idx[cluster] = i

# Print sentence
print("Sentence closest to the center of the cluster")
for idx in closest_idx:
    print(original_reviews[idx])
    print("Idx", idx, "Distance", alldistances[idx], "Score", scores[idx])
    print()



# PCA
pca = PCA(n_components=NUM_CLUSTERS).fit(X)
coords = pca.transform(X)
label_colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]
colors = [label_colors[i] for i in labels]
plt.scatter(coords[:, 0], coords[:, 1], c = colors)
centroids = kmeans.cluster_centers_
centroid_coords = pca.transform(centroids)
plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker="X", s = 200)
plt.show()
