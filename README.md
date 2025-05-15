Identifying Relevant Tweets Given a News Article
This project aims to identify tweets that are contextually relevant to a given news article using natural language processing and information retrieval techniques.

Developed under the guidance of Dr. Roshni Chakraborty
Project Group 7 –
Aditya Garg (2021IMT-005), Ayush Rakesh (2021IMT-018), Shivam Singh (2021IMT-091), Vinay Kokate (2021IMT-111)

Overview
We designed a pipeline to extract keywords from news articles and retrieve tweets that semantically or lexically match those keywords. The main techniques employed include:

KeyBERT and YAKE for keyword extraction

SIMCSE for semantic similarity

BM25 for lexical similarity

Pseudo Relevance Feedback to further enhance retrieval quality

 Key Components
 Keyword Extraction
KeyBERT: Extracts keywords using BERT-based embeddings and cosine similarity.

YAKE: A backup unsupervised method based on statistical patterns.

Chunking: Articles are split into 4-sentence chunks for better BERT performance.

Tweet Preprocessing
Filters only English tweets using langdetect

Removes URLs, RTs, emojis, and non-ASCII characters

Extracts hashtags and stores data in structured format

Similarity Computation
SIMCSE: Generates embeddings and ranks tweets based on cosine similarity

BM25: Calculates lexical similarity scores with query-term matching and IDF

Pseudo Relevance Feedback
Enhances top-k results by finding similar tweets to high-ranking ones

Combines content and hashtags into a semantic vector space

Evaluation Metrics
We evaluate tweet relevance using three metrics:

Keyword Match – Proportion of article keywords found in a tweet

Jaccard Similarity – Measures lexical overlap

Cosine Embedding Similarity – Captures semantic similarity

Results & Observations
Tweets scoring above 0.7–0.75 (normalized) were contextually relevant

SIMCSE outperforms BM25 in semantic similarity

BM25 is faster and useful for lexical overlap

Combining both methods improves robustness

Technologies Used
Python

KeyBERT, YAKE, SIMCSE

Sentence Transformers

BM25 (Rank-BM25)

NumPy, Scikit-learn, Langdetect
