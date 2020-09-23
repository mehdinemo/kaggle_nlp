import json
import re
import sys
from typing import List

import math
import pandas as pd
import requests
from tqdm import tqdm

from Infrastructure import ParallelManager
from LogManager import LogManager
from NLPInfrastructure import NGramManager, RuleManager


# region Entities
class Sentence:
    Id: int
    Text: str

    def __init__(self, unique_id, text):
        self.Id = unique_id
        self.Text = text


# endregion

class KeywordManager:
    def __init__(self, logger: LogManager, plm: ParallelManager, silent_mode: bool = False):
        self.silent_mode = silent_mode
        if not self.silent_mode:
            self.logger = logger
            self.reporter = tqdm
        else:
            self.reporter = fake_report

        self.ngm = NGramManager(silent_mode)
        self.rlm = RuleManager()
        self.plm = plm
        self.keyword_type_value = 1
        self.hashtag_type_value = 5

    def get_ngrams(self, sentences: List[Sentence], max_ngram_length):
        # ساخت tuple برای اجرا به صورت پارالل
        args = []
        for sentence in sentences:
            args.append((sentence.Id, sentence.Text, max_ngram_length, self.ngm))
        if not self.silent_mode:
            self.logger.info("Create NGrams for Messages", True)
        all_ngrams = self.plm.ParallelLoop(args, extract_ngrams)
        return all_ngrams

    def get_keywords_inclusion(self, sentences: List[Sentence], max_ngram_length, selected_ids: List[int] = None):
        # ساخت ngram از روی پیام ها
        ngrams_dic = {i: [] for i in range(1, max_ngram_length)}

        # ساخت tuple برای اجرا به صورت پارالل
        args = []
        for sentence in sentences:
            args.append((sentence.Id, sentence.Text, max_ngram_length, self.ngm))
        if not self.silent_mode:
            self.logger.info("Create NGrams for Messages", True)
        all_ngrams = self.plm.ParallelLoop(args, extract_ngrams)

        # جداسازی جملات با آی دی های وارد شده
        if selected_ids is not None:
            selected_sentences = [s.Id for s in sentences if s.Id in selected_ids]
            selected_sentences_ngrams = []
            for row in all_ngrams:
                sentence_id = row[0]
                ngrams = row[1]
                if sentence_id in selected_sentences:
                    selected_sentences_ngrams.append(ngrams)
        else:
            selected_sentences_ngrams = [ngrams[1] for ngrams in all_ngrams]

        # سرهم کردن خروجی
        for r in selected_sentences_ngrams:
            for l in range(1, max_ngram_length):
                ngrams_dic[l].extend(r[l])
        if not self.silent_mode:
            self.logger.info("Calculate Frequency of NGrams...")
        ngrams_frequency = self.ngm.get_words_frequency(ngrams_dic, max_ngram_length)
        for i in range(1, max_ngram_length):
            # فقط ngram های انتخاب می شوند که بیشتر از n بار تکرار شده اند
            ngrams_frequency[i] = {key: value for key, value in ngrams_frequency[i].items() if (value > 5)}
        if not self.silent_mode:
            self.logger.info("Merge Ngrams...")
        # شمول و همپوشانی
        merged_ngrams = self.ngm.merge_inclusions(ngrams_frequency, max_ngram_length)
        merged_ngrams = self.ngm.merge_overlaps(merged_ngrams, max_ngram_length)

        ngrams = {}
        for i in range(1, max_ngram_length):
            ngrams.update(merged_ngrams[i])

        # اعمال قوانین
        if not self.silent_mode:
            self.logger.info("Apply Rules on Ngrams...")
        ngrams = self.rlm.apply_rules_red(ngrams)
        ngrams = self.rlm.apply_rules_yellow(ngrams)
        ngrams = self.rlm.apply_rules_blue(ngrams)

        # نتایج نهایی
        final_selected_ngrams = list(ngrams.keys())
        # محاسبه حضور ngram های نهایی درون تمام پیام ها
        if not self.silent_mode:
            self.logger.info(
                "Calculate {0} NGram Inclusion for {1} Messages...".format(len(final_selected_ngrams), len(sentences)),
                True)

        # create allKeywords
        all_keywords = []
        for idx in self.reporter(range(len(sentences))):
            sentence = sentences[idx]
            ngrams = all_ngrams[idx][1]

            sentence_agg_ngrams = []
            for i in range(1, max_ngram_length):
                sentence_agg_ngrams.extend(ngrams[i])

            selected_ngrams = intersection(sentence_agg_ngrams, final_selected_ngrams)

            words = []
            for ngram in selected_ngrams:
                ngram_count_in_msg = sentence_agg_ngrams.count(ngram)
                if ngram_count_in_msg > 0:
                    words.append([sentence.Id, ngram, ngram_count_in_msg, self.keyword_type_value])
            all_keywords.extend(words)

        if not self.silent_mode:
            clear_last_line()

        if not self.silent_mode:
            self.logger.info("{0} Keywords Extracted. ".format(len(all_keywords)))

        df = pd.DataFrame(all_keywords, columns=['Id', 'Keyword', 'Frequency', 'Type'])
        return df

    def get_hashtags(self, sentences: List[Sentence]):
        if not self.silent_mode:
            self.logger.info("Extract Hashtags for Messages")
        pattern = r"#(\w+)"
        all_hashtags = []
        for sentence in sentences:
            if '#' in sentence.Text:
                text = sentence.Text.replace('\n', ' ').replace('\r', '')
                hashtags = re.findall(pattern, text, re.UNICODE)
                for h in hashtags:
                    all_hashtags.append(
                        [sentence.Id, '#{0}'.format(h), text.count('#{0}'.format(h)), self.hashtag_type_value])
        if not self.silent_mode:
            self.logger.info("{0} Hashtags Extracted. ".format(len(all_hashtags)))

        df = pd.DataFrame(all_hashtags, columns=['Id', 'Keyword', 'Frequency', 'Type'])
        return df

    def get_ner(self, sentences: List[Sentence], ner_server_address, ner_batch_size, selected_ids: List[int] = None):

        # جداسازی جملات با آی دی های وارد شده
        if selected_ids is not None:
            selected_sentences = [s for s in sentences if s.Id in selected_ids]
        else:
            selected_sentences = sentences

        msgs_batches = []
        batch_size = ner_batch_size
        batch_count = int(len(selected_sentences) / batch_size) if (len(selected_sentences) / batch_size == 0) else int(
            len(selected_sentences) / batch_size) + 1
        for i in range(batch_count):
            if i == batch_count - 1:
                msgs_batches.append(selected_sentences[i * batch_size:])
            else:
                msgs_batches.append(selected_sentences[i * batch_size:(i + 1) * batch_size])

        ner_items = [(msg, ner_server_address) for msg in msgs_batches]
        if not self.silent_mode:
            self.logger.info("Calculate NER for Messages".format(batch_count))
        result = self.plm.ParallelLoop(ner_items, get_message_ner)
        final_result = []
        for batch in result:
            final_result.extend(batch)
        if not self.silent_mode:
            self.logger.info("{0} NER Results Retrieved. ".format(len(final_result)))

        df = pd.DataFrame(final_result, columns=['Id', 'Keyword', 'Frequency', 'Type'])
        return df

    def calculate_grouped(self, keywords):
        user_used_word = keywords.groupby(['Keyword', 'UserId']).agg({'Frequency': 'sum', 'Type': 'max'}).reset_index()
        user_used_word['Frequency2'] = user_used_word['Frequency']
        user_used_word['Frequency2'] = user_used_word['Frequency2'].apply(lambda x: math.log(x + 1))

        grouped_df = user_used_word.groupby(['Keyword']).agg({'Frequency': 'sum',
                                                              'Frequency2': 'sum',
                                                              'UserId': 'count',
                                                              'Type': 'max'}).reset_index()
        grouped_df.rename(columns={"Frequency": "sum", "Frequency2": "fsum"}, inplace=True)
        fsum = grouped_df["fsum"].sum(axis=0, skipna=True)
        sum = grouped_df["sum"].sum(axis=0, skipna=True)

        grouped_df["sum_normal"] = grouped_df["sum"].apply(lambda x: x / sum)
        grouped_df["fsum_normal"] = grouped_df["fsum"].apply(lambda x: x / fsum)

        return grouped_df

    def filter_keywords_by_frequency(self, keywords: pd.DataFrame, min_msg_count: int = 5,
                                     min_user_count: int = 3) -> pd.DataFrame:
        # فیلتر تعداد پیام
        # group by Keyword on Id
        grouped = keywords.groupby(['Keyword']).agg({'Id': 'count'}).reset_index()
        removed_keywords = grouped[grouped["Id"] < min_msg_count]
        min_msg_filtered_keywords = keywords[~ keywords["Keyword"].isin(removed_keywords["Keyword"].values)]

        # فیلتر تعداد user
        # group by Keyword
        # count unique users
        grouped = min_msg_filtered_keywords.groupby(['Keyword'])['UserId'].nunique().reset_index()
        removed_keywords = grouped[grouped["UserId"] < min_user_count]
        min_user_filtered_keywords = \
            min_msg_filtered_keywords[~ min_msg_filtered_keywords["Keyword"].isin(removed_keywords["Keyword"].values)]

        return min_user_filtered_keywords

    def filter_keywords(self, keywords):
        drop_indexes = []
        for idx, keyword_tuple in enumerate(keywords.values):
            k = keyword_tuple[1]
            if ('\ufffe' in k.lower()) | ('\uffff' in k.lower()) | ('\ufffd' in k.lower()):
                drop_indexes.append(idx)

        keywords.drop(keywords.index[drop_indexes], inplace=True)
        return keywords


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def clear_last_line():
    sys.stdout.write("\033[F")  # Cursor up one line
    sys.stdout.write("\033[K")  # Clear to the end of line


# این تابع برای غیر فعال کردن tqdm استفاده خواهد شد
def fake_report(inp):
    return inp


# region ParallelFunctions
def extract_ngrams(arg):
    msg_id = arg[0]
    text = arg[1]
    max_length = arg[2]
    ngm = arg[3]
    res = ngm.normalize_tokenize_extract_ngrams(text, max_length)
    return [msg_id, res]


def get_message_ner(args):
    sentences = args[0]
    ner_server_address = args[1]
    headers = {'Content-type': 'application/json'}

    # ساخت بدنه request
    body = {}
    body['messages'] = []
    for idx, sentence in enumerate(sentences):
        if sentence.Text.strip() == "":
            continue
        sentence_text = sentence.Text.replace('\'', '').replace('\"', '').replace('{', '').replace('}', '')
        body['messages'].append({"Id": str(sentence.Id), "message": sentence_text})

    body = json.dumps(body, ensure_ascii=False, indent=4)

    resp = requests.post(ner_server_address, headers=headers, data=body.encode('utf-8'), timeout=None,
                         verify=False)
    resp = json.loads(resp.text)

    # تبدیل به لیست
    result = []
    for r in resp:
        # دریافت message در لیست messages ورودی
        for sentence in sentences:
            if (int(sentence.Id) == int(r[2])):
                ner_word = r[0].replace('\\', ' ')
                try:
                    ner_type = int(r[1])
                except:
                    break
                result_obj = [sentence.Id, ner_word, sentence.Text.count(ner_word), ner_type]
                if result_obj not in result:
                    result.append(result_obj)
                break
    return result

# endregion
