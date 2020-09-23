from tqdm import tqdm
import itertools
import codecs
import sys

from NLPInfrastructure import SentenceNormalizer, SentenceTokenizer


class NGramManager:
    def __init__(self, silent_mode: bool = False):
        self.silent_mode = silent_mode
        if not self.silent_mode:
            self.reporter = tqdm
        else:
            self.reporter = fake_report

        self.normalizer = SentenceNormalizer()
        self.tokenizer = SentenceTokenizer()

    def extract_ngrams(self, text, ngram_length):
        ngrams_dict = {}
        for i in range(1, ngram_length):
            ngrams = self.__generate_ngrams(text, i)
            ngrams_dict[i] = [ngram.strip() for ngram in ngrams]
        return ngrams_dict

    def normalize_tokenize_extract_ngrams(self, text, ngram_length):
        text = self.normalizer.replace_urls(text)
        text = self.normalizer.replace_emails(text)
        text = self.normalizer.replace_usernames(text)
        text = self.normalizer.replace_hashtags(text)
        text = self.normalizer.replace_phone_numbers(text)
        text = self.normalizer.replace_emoji(text)

        token_text = self.tokenizer.tokenize_text_new(text)

        ngrams_dict = {}
        for i in range(1, ngram_length):
            ngrams_dict[i] = []
            for token in token_text:
                ngrams = self.__generate_ngrams(token[0], i)
                f_ngrams = []
                for ngram in ngrams:
                    if (not '\ufffe' in ngram.lower()) & \
                            (not '\uffff' in ngram.lower()) & \
                            (not '\ufffd' in ngram.lower()):
                        f_ngrams.append(ngram.strip())

                ngrams_dict[i].extend(f_ngrams)

        return ngrams_dict

    def extract_ngrams_as_list(self, text, ngram_length):
        ngrams_result = []
        for i in range(1, ngram_length):
            ngrams = self.__generate_ngrams(text, i)
            ngrams_result.extend([ngram.strip() for ngram in ngrams])
        return ngrams_result

    def get_words_frequency(self, ngrams_dic, ngram_length):
        words_frequency = {}
        for i in range(1, ngram_length):
            ngrams = ngrams_dic[i]
            w_freq = {key: len(list(group)) for key, group in itertools.groupby(sorted(ngrams))}
            words_frequency[i] = w_freq
        return words_frequency

    def merge_inclusions(self, words_frequency, ngram_length):

        result = {i: {} for i in range(1, ngram_length)}
        for i in range(1, ngram_length - 1):

            words_smaller = list(words_frequency[i].keys())
            words_bigger = list(words_frequency[i + 1].keys())

            for smaller_index in self.reporter(range(len(words_smaller))):

                smaller = words_smaller[smaller_index]
                is_merged = False

                for bigger_index in range(len(words_bigger)):
                    bigger = words_bigger[bigger_index]
                    smaller_frequency = words_frequency[i][smaller]
                    bigger_frequency = words_frequency[i + 1][bigger]

                    if ((bigger_frequency / smaller_frequency) > 0.8):
                        if self._check_sub_ngram(smaller, bigger):
                            is_merged = True
                            break

                if not is_merged:
                    result[i][smaller] = words_frequency[i][smaller]
            # پاک کردن tqdm
            if not self.silent_mode:
                clear_last_line()

        result[ngram_length - 1].update(words_frequency[ngram_length - 1])
        return result

    def merge_overlaps(self, words_frequency, ngram_length):
        result = {i: {} for i in range(1, ngram_length)}
        gram_length = 2
        words = list(words_frequency[gram_length].keys())

        for i in self.reporter(range(len(words))):
            w1 = words[i]
            w1_parts = w1.split(' ')
            w1_frequency = words_frequency[gram_length][w1]
            is_merged = False
            for j in range(len(words)):
                w2 = words[j]
                w2_parts = w2.split(' ')
                w2_frequency = words_frequency[gram_length][w2]

                if len(w1_parts) < 2:
                    continue

                # check overlap
                if w1_parts[1] == w2_parts[0]:
                    # check frequency
                    if ((w1_frequency / w2_frequency) > 0.8) & ((w1_frequency / w2_frequency) < 1.2):
                        # check 3gram exists
                        new_gram = [w1_parts[0], w1_parts[1], w2_parts[1]]
                        new_gram = ' '.join(new_gram)
                        if new_gram in words_frequency[gram_length + 1]:
                            is_merged = True
                            break
            if not is_merged:
                result[gram_length][w1] = words_frequency[gram_length][w1]

        result[1].update(words_frequency[1])
        result[3].update(words_frequency[3])

        # پاک کردن tqdm
        if not self.silent_mode:
            clear_last_line()

        return result

    def _check_sub_ngram(self, smaller, bigger):
        if not smaller in bigger:
            return False

        smaller_parts = smaller.split(' ')
        smaller_parts = [p for p in smaller_parts if ((p != " ") & (p != ""))]
        bigger_parts = bigger.split(' ')
        bigger_parts = [p for p in bigger_parts if ((p != " ") & (p != ""))]

        if len(smaller_parts) > len(bigger_parts):
            return False

        sub_ngrams = self.__generate_ngrams(bigger, len(smaller_parts))

        if smaller in sub_ngrams:
            return True
        return False

    def __n_grams(self, seq, n=1):
        shift_token = lambda i: (el for j, el in enumerate(seq) if j >= i)
        shifted_tokens = (shift_token(i) for i in range(n))
        tuple_ngrams = zip(*shifted_tokens)
        return tuple_ngrams

    def __generate_ngrams(self, s, n):
        tokens = [token for token in s.split(" ") if ((token != " ") & (token != ""))]
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def __clean_sentence(self, text):
        return self.normalizer.normalize_sentence(text)


def clear_last_line():
    sys.stdout.write("\033[F")  # Cursor up one line
    sys.stdout.write("\033[K")  # Clear to the end of line


# این تابع برای غیر فعال کردن tqdm استفاده خواهد شد
def fake_report(inp):
    return inp


if __name__ == '__main__':
    db = NGramManager('./result/')
    msg = "مقامات ارشد کابینه آمریکا : حملات علیه کشتی های تجاری در نزدیکی امارات متحده عربی و خط لوله های نفتی عربستان سعودی توسط ایران برنامه ریزی و هدایت شده است. #trumpabolishthemullahs"
    res = db.normalize_tokenize_extract_ngrams(msg, 4)
    print('Done')
