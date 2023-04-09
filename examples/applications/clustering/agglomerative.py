from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

dataset_path = "reviews.tsv"
max_corpus_size = 20000  # We limit our corpus to only the first 20k questions

# Get all unique sentences from the file
corpus_sentences = set()
with open(dataset_path, encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        corpus_sentences.add(row['headline'])
        if len(corpus_sentences) >= max_corpus_size:
            break

corpus_sentences = list(corpus_sentences)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_sentences)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.toarray())

kmeans = KMeans(n_clusters=7, n_init=20)
kmeans.fit(X)

fig, ax = plt.subplots(figsize=(10, 8))

clusters = {}
for i, label in enumerate(kmeans.labels_):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(corpus_sentences[i])

for label in clusters:
    indices = [i for i, x in enumerate(kmeans.labels_) if x == label]
    ax.scatter(X_2d[indices, 0], X_2d[indices, 1], label=f'Cluster {label}')

ax.legend()
plt.show()

for label, sentences in clusters.items():
    print("Cluster ", label, ":")
    for sentence in sentences[0:5]:
        print(" - ", sentence)
    print("\t", "...")
    for sentence in sentences[-5:]:
        print(" - ", sentence)
