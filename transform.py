from nltk.tokenize import sent_tokenize, word_tokenize


class TextTransform(object):
    def __init__(self, tokenizer=word_tokenize, pre_process=str.lower):
        self.pre_process = pre_process
        self.tokenizer = tokenizer
    
    def __call__(self, text):
        if self.pre_process is not None:
            text = self.pre_process(text)
        tokens = self.tokenizer(text)
        return tokens

transform = TextTransform(pre_process=None)