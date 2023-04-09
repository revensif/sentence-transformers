"""
This is a more complex example on performing clustering on large scale dataset.

This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.

A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.

The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).

In this example, we download a large set of questions from Quora and then find similar questions in this set.
"""

import csv
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer, util

# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer('all-MiniLM-L6-v2')
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

print("Encode the corpus. This might take a while")
corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(corpus_embeddings)

print("Start clustering")
start_time = time.time()

kmeans = KMeans(n_clusters=7, n_init=20)
kmeans.fit(corpus_embeddings)

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

# Print for all clusters the top 3 and bottom 3 elements
for label, sentences in clusters.items():
    print("Cluster ", label, ":")
    for sentence in sentences[0:5]:
        print(" - ", sentence)
    print("\t", "...")
    for sentence in sentences[-5:]:
        print(" - ", sentence)

print("Start clustering")
start_time = time.time()

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(corpus_embeddings, min_community_size=100, threshold=0.75)

print("Clustering done after {:.2f} sec".format(time.time() - start_time))

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[sentence_id])
