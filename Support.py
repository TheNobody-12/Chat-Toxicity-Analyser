from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import re

class TextPreprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def remove_emoji(self, string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def preprocess_text(self, text):
        # Converting Object Data type into String Datatype
        text = str(text)
        # Removal of Punctuation - !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`
        text = re.sub(r'[^\w\s]', '', text)
        # Remove URLs
        text = self.remove_urls(text)
        # Removing Emojis
        text = self.remove_emoji(text)
        # Lower casing the Text
        text = text.lower()
        # Stopwords removing
        text = [word for word in word_tokenize(text) if word not in self.stop_words]
        # # Removing Frequent Words
        # cnt = Counter()
        # for word in text:
        #     cnt[word] += 1
        # FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
        text = " ".join([word for word in text])
        # Tokenization
        text = word_tokenize(text)
        # Stemming and Lemmatization
        text = [self.lemmatizer.lemmatize(word) for word in text]
        # text = [self.stemmer.stem(word) for word in text]

        return text


