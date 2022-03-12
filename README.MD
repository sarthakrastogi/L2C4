# L2C4

## A Language Lending Itself: Mapping Clusters of Contextually Close Cognates in Indo-European Languages

Paper by Sarthak Rastogi



**1.0 Data Ingestion.ipynb:** Loading word embeddings into dictionaries and pickling.

**1.1 Preprocessing.ipynb:**

* POS tagging for removing proper nouns like countries, nationalities, and brands.
* Removing special characters
* Lemmatisation
* Choosing most frequently used words
* Removing names using a list of common names

**2.1 - 2.5 Translation:** Attempts at translation of word embeddings. Finally, uploading the embedding keys as a document to Yandex Translate was found to be more efficient.

**3.0 Transliteration.ipynb:** Transliterating words in Indian languages from their native scripts to the Roman script. Removing definitive (le, la, l’ and les) and partitive (du, de la, des, de l', de, d') articles from French words.

**4.0 Phonetic Matching.ipynb:** Calculating the Double Metaphone encodings for the words and matching words with similar encodings, i.e., cognates, onto a new embedding space

**5.0 Clustering Experiments.ipynb:** Experimenting with hyperparameter values of various clustering algorithms.

**clustering.py:** Contains functions for running the above experiments.

**5.1 Clustering Optimal.ipynb:** Clustering on the language pair embeddings using the optimal hyperparameter values obtained in the previous notebook.

**clustering_optimal_algos.py:** Contains functions for optimal clustering.