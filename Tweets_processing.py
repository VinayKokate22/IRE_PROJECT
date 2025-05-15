import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  

def preprocess_english_tweets_with_hashtags(tweets_file):
    with open(tweets_file, 'r', encoding='utf-8') as file:
        raw_tweets = file.readlines()
    abc=1
    cleaned_tweets = []

    for idx, tweet in enumerate(raw_tweets, 1):
        parts = tweet.strip().split("*,,,*")
        abc=abc+1
        if(abc==1000):
            break
        if len(parts) >= 7:
            tweet_text = parts[2]
            hashtag_field = parts[6]

            raw_tags = re.findall(r'\*([^*]+)\*', hashtag_field)
            hashtags_cleaned = [tag.lower().strip() for tag in raw_tags if tag.strip()]

            tweet_text = re.sub(r'http\S+', '', tweet_text)

            try:
                if detect(tweet_text) != 'en':
                    continue
            except:
                continue 

            # Lowercase
            content = tweet_text.lower()

            content = re.sub(r'\\ud[0-9a-f]{3}', '', content)  
            content = re.sub(r'[^a-z0-9\s]', '', content) 

            for tag in hashtags_cleaned:
                content = re.sub(r'\b' + re.escape(tag) + r'\b', '', content)

            content = re.sub(r'\s+', ' ', content).strip()

            if content:  
                cleaned_tweets.append({
                    'content': content,
                    'hashtags': hashtags_cleaned
                })

    return cleaned_tweets

tweets_file_path = r'D:\Telegram Desktop\Ire_project\Ire_project\tweets.txt'
cleaned_tweets = preprocess_english_tweets_with_hashtags(tweets_file_path)

content_file_path = r'D:\Telegram Desktop\Ire_project\Ire_project\cleaned_tweets_content.txt'
hashtags_file_path = r'D:\Telegram Desktop\Ire_project\Ire_project\cleaned_tweets_hashtags.txt'

with open(content_file_path, 'w', encoding='utf-8') as content_file:
    for tweet in cleaned_tweets:
        content_file.write(tweet['content'] + '\n')

with open(hashtags_file_path, 'w', encoding='utf-8') as hashtags_file:
    for tweet in cleaned_tweets:
        hashtags_file.write(' '.join(tweet['hashtags']) + '\n')

print("Cleaned English Tweets (First 20):\n")
for i, tweet in enumerate(cleaned_tweets[:20], 1):
    print(f"Tweet {i}:")
    print("Content :", tweet['content'])
    print("Hashtags:", tweet['hashtags'])
    print()

print(f"\nFiles saved:\n- {content_file_path}\n- {hashtags_file_path}")
