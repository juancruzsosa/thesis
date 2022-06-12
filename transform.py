import string
from nltk.tokenize import sent_tokenize, word_tokenize

table = str.maketrans(string.punctuation," " * len(string.punctuation))

def preprocess(text: str):
    text = text.translate(table)
    text = text.lower()
    return text

class TextTransform(object):
    def __init__(self, tokenizer=word_tokenize, pre_process=preprocess):
        self.pre_process = pre_process
        self.tokenizer = tokenizer
    
    def __call__(self, text):
        if self.pre_process is not None:
            text = self.pre_process(text)
        tokens = self.tokenizer(text)
        return tokens