import re
import nltk
import time
import torch
import numpy as np
from tqdm import tqdm
from keybert import KeyBERT
from yake import KeywordExtractor
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

nltk.download('punkt')

# Load SIMCSE Model
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
simcse_model = SentenceTransformer('princeton-nlp/sup-simcse-roberta-base')

def chunk_text(text, chunk_size=4):
    sentences = sent_tokenize(text)
    return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

def get_yake_keywords(text):
    extractor = KeywordExtractor(lan="en", n=2, top=10)
    return [kw for kw, score in extractor.extract_keywords(text)]

def extract_articles_with_keywords(file_path):
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    article_pattern = r'(?:\{14})?-----\s(.?)\s-----\s*(.?)\s-----\s*(.?)(?=\{14}-----|$)'
    matches = re.findall(article_pattern, text, re.DOTALL)

    articles = []
    for headline, date, content in matches:
        article = {
            'headline': headline.strip(),
            'date': date.strip(),
            'content': content.strip(),
        }
        chunks = chunk_text(article['content'], chunk_size=4)
        all_keywords = []
        for chunk in chunks:
            if len(chunk.strip()) < 40:
                continue
            chunk_keywords = kw_model.extract_keywords(
                chunk,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=10
            )
            filtered = [kw for kw, score in chunk_keywords if score > 0.4]
            all_keywords.extend(filtered)
        if not all_keywords:
            all_keywords = get_yake_keywords(article['content'])
        seen = set()
        deduped_keywords = []
        for kw in all_keywords:
            if kw not in seen:
                deduped_keywords.append(kw)
                seen.add(kw)
        article['keywords'] = deduped_keywords[:10]
        articles.append(article)
    return articles

def read_preprocessed_tweets(tweets_file):
    tweets = []
    current_tweet = {}
    with open(tweets_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Content:"):
                current_tweet['content'] = line[len("Content:"):].strip()
            elif line.startswith("Hashtags:"):
                current_tweet['hashtags'] = line[len("Hashtags:"):].strip()
                tweets.append(current_tweet)
                current_tweet = {}
    return tweets

def simcse_encode(texts, batch_size=32, device='cpu'):
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="SIMCSE embeddings"):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            model_output = model(**encoded)
            embeddings_batch = mean_pooling(model_output, encoded['attention_mask'])
            embeddings.append(embeddings_batch.cpu())
    return torch.cat(embeddings, dim=0)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def find_relevant_tweets_simcse(tweets, keywords):
    def get_text(tweet):
        if tweet['hashtags'].lower() != 'none':
            return tweet['content'] + ' ' + ' '.join(f"#{tag.strip()}" for tag in tweet['hashtags'].split())
        return tweet['content']

    tweet_texts = [get_text(tweet) for tweet in tweets]
    keyword_text = ' '.join(keywords)
    print("\n→ Generating SIMCSE embeddings (SIMCSE only)...")
    tweet_embeddings = simcse_encode(tweet_texts)
    keyword_embedding = simcse_encode([keyword_text])
    cosine_similarities = torch.nn.functional.cosine_similarity(keyword_embedding, tweet_embeddings)
    simcse_scores = cosine_similarities.numpy()
    scored_tweets = list(zip(tweets, simcse_scores))
    return sorted(scored_tweets, key=lambda x: x[1], reverse=True)

def find_relevant_tweets_bm25(tweets, keywords):
    def get_text(tweet):
        if tweet['hashtags'].lower() != 'none':
            return tweet['content'] + ' ' + ' '.join(f"#{tag.strip()}" for tag in tweet['hashtags'].split())
        return tweet['content']

    tweet_texts = [get_text(tweet) for tweet in tweets]
    keyword_text = ' '.join(keywords)
    print("\n→ Calculating BM25 scores (BM25 only)...")
    tokenized_tweets = [text.lower().split() for text in tqdm(tweet_texts, desc="Tokenizing tweets")]
    bm25 = BM25Okapi(tokenized_tweets)
    bm25_scores = bm25.get_scores(keyword_text.lower().split())
    bm25_max_score = max(bm25_scores)
    bm25_normalized = np.array(bm25_scores) / bm25_max_score
    scored_tweets = list(zip(tweets, bm25_normalized))
    return sorted(scored_tweets, key=lambda x: x[1], reverse=True)

def expand_tweets_with_similarity(top_tweets, all_tweets, top_n=10):
    def enrich(tweet):
        text = tweet['content'].lower()
        if tweet['hashtags'].lower() != "none":
            text += ' ' + ' '.join(f"#{tag.strip()}" for tag in tweet['hashtags'].split())
        return text

    enriched_all = [enrich(tweet) for tweet in all_tweets]
    enriched_top = [enrich(tweet) for tweet, _ in top_tweets[:top_n]]
    vectorizer = TfidfVectorizer(stop_words='english')
    all_vectors = vectorizer.fit_transform(enriched_all)
    top_vectors = vectorizer.transform(enriched_top)
    similarity_matrix = cosine_similarity(top_vectors, all_vectors)
    avg_similarity = np.mean(similarity_matrix, axis=0)
    scored_expansion = list(zip(all_tweets, avg_similarity))
    scored_expansion.sort(key=lambda x: x[1], reverse=True)
    top_texts_set = set(enrich(tweet) for tweet, _ in top_tweets[:top_n])
    expanded_tweets = [(tweet, score) for tweet, score in scored_expansion if enrich(tweet) not in top_texts_set]
    return expanded_tweets[:10]

def evaluate_keyword_coverage(tweets, keywords):
    coverage = []
    for tweet in tweets:
        text = tweet['content']
        if tweet['hashtags'].lower() != 'none':
            text += ' ' + tweet['hashtags']
        matched = [kw for kw in keywords if kw.lower() in text.lower()]
        coverage.append(len(matched))
    return sum(coverage) / len(tweets)

def jaccard_similarity(tweet_text, keywords):
    tweet_words = set(tweet_text.lower().split())
    keyword_set = set(kw.lower() for kw in keywords)
    intersection = tweet_words & keyword_set
    union = tweet_words | keyword_set
    return len(intersection) / len(union) if union else 0.0

def evaluate_jaccard_coverage(tweets, keywords):
    scores = []
    for tweet in tweets:
        text = tweet['content']
        if tweet['hashtags'].lower() != "none":
            text += ' ' + tweet['hashtags']
        scores.append(jaccard_similarity(text, keywords))
    return sum(scores) / len(scores)

def average_cosine_similarity(tweets, keywords, model, tokenizer, device='cpu'):
    texts = [tweet['content'] + ' ' + tweet['hashtags'] if tweet['hashtags'].lower() != 'none' else tweet['content'] for tweet in tweets]
    embeddings = simcse_encode(texts, device=device)
    keyword_embedding = simcse_encode([' '.join(keywords)], device=device)
    similarities = torch.nn.functional.cosine_similarity(keyword_embedding, embeddings).numpy()
    return np.mean(similarities)

def compare_methods(simcse_results, bm25_results, keywords, model, tokenizer, device='cpu'):
    top_simcse_tweets = [t for t, _ in simcse_results[:10]]
    top_bm25_tweets = [t for t, _ in bm25_results[:10]]
    simcse_coverage = evaluate_keyword_coverage(top_simcse_tweets, keywords)
    bm25_coverage = evaluate_keyword_coverage(top_bm25_tweets, keywords)
    simcse_jaccard = evaluate_jaccard_coverage(top_simcse_tweets, keywords)
    bm25_jaccard = evaluate_jaccard_coverage(top_bm25_tweets, keywords)
    simcse_sim = average_cosine_similarity(top_simcse_tweets, keywords, model, tokenizer, device)
    bm25_sim = average_cosine_similarity(top_bm25_tweets, keywords, model, tokenizer, device)
    print("\n Evaluation Summary:")
    print(f" Avg Keyword Match:       SIMCSE = {simcse_coverage:.2f}, BM25 = {bm25_coverage:.2f}")
    print(f" Avg Jaccard Similarity:  SIMCSE = {simcse_jaccard:.2f}, BM25 = {bm25_jaccard:.2f}")
    print(f" Avg Embedding Similarity:SIMCSE = {simcse_sim:.4f}, BM25 = {bm25_sim:.4f}")
    best_score = max((simcse_coverage, 'SIMCSE'), (bm25_coverage, 'BM25'))[1]
    print(f"\n Based on keyword match, {best_score} appears more relevant.")

# MAIN EXECUTION
start_time = time.time()
file_path = r'D:\Telegram Desktop\Ire_project\Ire_project\us_newsarticle_August_1.txt'
tweets_file_path = r'D:\Telegram Desktop\Ire_project\Ire_project\tweet_clean.txt'
articles = extract_articles_with_keywords(file_path)
tweets = read_preprocessed_tweets(tweets_file_path)
first_article_keywords = articles[1]['keywords']
print(f"\nEvaluating tweets for Article: {articles[1]['headline']}")
print(f"Keywords: {first_article_keywords}")
relevant_tweets_simcse = find_relevant_tweets_simcse(tweets, first_article_keywords)
relevant_tweets_bm25 = find_relevant_tweets_bm25(tweets, first_article_keywords)
print("\nTop 10 Relevant Tweets (SIMCSE):")
for tweet, score in relevant_tweets_simcse[:10]:
    print(f"Tweet: {tweet['content']} | Hashtags: {tweet['hashtags']} | SIMCSE Score: {score:.4f}")
print("\nTop 10 Relevant Tweets (BM25):")
for tweet, score in relevant_tweets_bm25[:10]:
    print(f"Tweet: {tweet['content']} | Hashtags: {tweet['hashtags']} | BM25 Score: {score:.4f}")
compare_methods(relevant_tweets_simcse, relevant_tweets_bm25, first_article_keywords, model, tokenizer)
expanded_tweets = expand_tweets_with_similarity(relevant_tweets_simcse, tweets)
print("\nExpanded Tweets via Similarity:")
for tweet, score in expanded_tweets:
    print(f"Tweet: {tweet['content']} | Similarity Score: {score:.4f}")
print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")