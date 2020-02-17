from collections import Counter, defaultdict
from nltk.tokenize import sent_tokenize, RegexpTokenizer


def remove_unknown_words(text, counter):
    for i, word in enumerate(text):
        if word not in counter:
            text[i] = '<unk>'
    return text


def tokenize_text(text):
    word_tokenizer = RegexpTokenizer(r'\w+')
    tokenized_text = []
    for sentence in sent_tokenize(text):
        tokenized_text.append('<start>')
        tokenized_text.extend(word_tokenizer.tokenize(sentence.lower()))
        tokenized_text.append('<end>')
    return tokenized_text


def remove_below_threshold(text, count_threshold):
    counter = Counter(text)
    for i, word in enumerate(text):
        if counter[word] < count_threshold:
            text[i] = '<unk>'
    counter = {word: count for word, count in counter.items() if count >= count_threshold}
    return text, counter


def build_n_grams(text, count_threshold, n):
    text, counter = remove_below_threshold(tokenize_text(text), count_threshold)
    n_grams = defaultdict(lambda: defaultdict(int))
    n_minus_one_gram_counts = defaultdict(int)
    for i in range(len(text) - n + 1):
        n_grams[tuple(text[i:i + n - 1])][text[i + n - 1]] += 1
        n_minus_one_gram_counts[tuple(text[i:i + n - 1])] += 1
    n_minus_one_gram_counts[tuple(text[len(text) - n + 1:])] += 1
    return n_grams, n_minus_one_gram_counts, counter


class LanguageModel:
    def __init__(self, n_gram=2):
        self.n = n_gram
        self.n_grams = None
        self.n_minus_one_gram_counts = None
        self.counter = None

    def build_model(self, text, count_threshold=4):
        self.n_grams, self.n_minus_one_gram_counts, self.counter = build_n_grams(text, count_threshold, self.n)

    def calculate_proba(self, sentence):
        sentence = remove_unknown_words(tokenize_text(sentence), self.counter)
        proba = 1.0
        for i in range(len(sentence) - self.n + 1):
            numerator = self.n_grams[tuple(sentence[i:i + self.n - 1])][sentence[i + self.n - 1]]
            denominator = self.n_minus_one_gram_counts[tuple(sentence[i:i + self.n - 1])]
            proba *= numerator / denominator
        return proba
