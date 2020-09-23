import codecs
import re

from .NormalizingMapping import list as separators


class SentenceTokenizer:
    def __init__(self):
        self.separators = separators
        self.separators = [a.strip() for a in self.separators]
        self.paragraph_splitter = '\uFFFE'
        self.sentence_splitter = '\uFFFF'
        self.splitter = '\uFFFD'

    def tokenize_text(self, text: str):
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')

        for s in self.separators:
            text = text.replace(s, '.')

        return text.split('.')

    def tokenize_text_new(self, text: str):
        # تبدیل ورودی به پاراگراف ها
        text = text.replace('\n\n', self.paragraph_splitter)
        text = text.replace('\n', self.paragraph_splitter)
        paragraphs = text.split(self.paragraph_splitter)

        sentences = []
        for p in paragraphs:
            if p.strip() == '':
                continue

            normalized_sentence = self._replace_splitter(p.strip())
            normalized_sentence = self._replace_consecutive_spaces(normalized_sentence)
            for s in normalized_sentence.split(self.sentence_splitter):
                if s.strip() == '':
                    continue
                sentences.append(s.split(self.splitter))

        return sentences

    def _replace_splitter(self, text):
        # جایگذاری علایم سجاوندی
        splitter = self.sentence_splitter

        def matched(obj):
            return splitter

        text = text.lower()

        p1 = re.compile(r'[!\.\?⸮؟/,;:\\،؛]+', re.UNICODE)
        result = p1.sub(matched, text)

        # جایگذاری splitter های نرمالگر
        for s in self.separators:
            result = result.replace(s.lower(), self.sentence_splitter)

        return result

    def _replace_consecutive_spaces(self, text: str):
        def matched(obj):
            return ' '

        text = text.lower()

        p1 = re.compile(r'[ ]{2,}', re.UNICODE)
        result = p1.sub(matched, text)
        return result


if __name__ == "__main__":
    st = SentenceTokenizer()
    print("Done!")
