import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stemmer = PorterStemmer()


def preprocess(text):
    # 1. Canonical string + lowercase for case-insensitive matching
    text = str(text).lower()
    # 2. Word tokens only (letters/digits/underscore per \w+)
    tokens = re.findall(r"\w+", text)
    # 3–4. Remove common function words and single-character noise
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    # 5. Stem so inflected forms share one index type
    return [stemmer.stem(t) for t in tokens]
