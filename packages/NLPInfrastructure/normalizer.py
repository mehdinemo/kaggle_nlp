import re
import regex


class SentenceNormalizer:
    def __init__(self):
        self.domains = "(?:gl|com|ir|org|net|edu|info|me|ac|name|biz|co|pro|ws|asia|mobi|tel|eu|in|ru|tv|cc|es|de|ca|mn|bz|uk|us|au)"

        # region static Character Mapping
        self._from_characters = ['،', '(', ')', '\t', '\u200c', ' ', '0', '1', '۱', '١', '2', '۲', '٢', '3', '۳', '٣',
                                 '4', '۴', '٤', '5', '۵', '٥', '6', '۶', '٦', '7', '۷', '٧', '8', '۸', '٨', '9', '۹',
                                 '٩', '.', ':', '%', '/', '@', '#', '_', 'a', 'A', 'b', 'B', 'c', 'C', 'D', 'd', 'e',
                                 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M',
                                 'n', 'N', 'o', 'O', 'p', 'P', 'Q', 'q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'v',
                                 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z', 'ء', 'ﺀ', 'ٱ', 'ا', 'ﺍ', 'ﺎ', 'إ', 'ﺇ',
                                 'أ', 'ﺃ', 'ﺄ', 'آ', 'ﺁ', 'ب', 'ﺑ', 'ﺒ', 'ﺏ', 'ﺐ', 'پ', 'ﭘ', 'ﭙ', 'ة', 'ﺔ', 'ﺓ', 'ت',
                                 'ﺘ', 'ﺖ', 'ﺗ', 'ﺕ', 'ٺ', 'ث', 'ﺛ', 'ﺜ', 'ﺚ', 'ج', 'ﺟ', 'ﺠ', 'ﺝ', 'ﺞ', 'چ', 'ﭼ', 'ﭽ',
                                 'ﭻ', 'ح', 'ﺣ', 'ﺤ', 'ﺡ', 'ﺢ', 'خ', 'ﺧ', 'ﺨ', 'ﺦ', 'ﺥ', 'د', 'ﺩ', 'ﺪ', 'ذ', 'ﺬ', 'ﺫ',
                                 'ر', 'ﺭ', 'ﺮ', 'ز', 'ﺯ', 'ﺰ', 'ژ', 'ﮋ', 'ﮊ', 'س', 'ﺳ', 'ﺴ', 'ﺲ', 'ﺱ', 'ش', 'ﺷ', 'ﺸ',
                                 'ﺶ', 'ﺵ', 'ص', 'ﺻ', 'ﺼ', 'ﺺ', 'ﺹ', 'ض', 'ﻀ', 'ﺿ', 'ﺽ', 'ﺾ', 'ط', 'ﻃ', 'ﻄ', 'ﻂ', 'ﻁ',
                                 'ظ', 'ﻈ', 'ﻇ', 'ﻆ', 'ع', 'ﻋ', 'ﻌ', 'ﻊ', 'ﻉ', 'غ', 'ﻏ', 'ﻐ', 'ﻍ', 'ﻎ', 'ف', 'ﻓ', 'ﻔ',
                                 'ﻒ', 'ﻑ', 'ق', 'ﻗ', 'ﻘ', 'ﻖ', 'ﻕ', 'ك', 'ﻛ', 'ﻜ', 'ﻙ', 'ﻚ', 'ک', 'ﮐ', 'ﮑ', 'ﮏ', 'ﮎ',
                                 'ڪ', 'گ', 'ﮔ', 'ﮕ', 'ﮓ', 'ﮒ', 'ل', 'ﻝ', 'ﻞ', 'ڵ', 'ﻼ', 'ﻻ', 'م', 'ﻣ', 'ﻤ', 'ﻢ', 'ﻡ',
                                 'ن', 'ﻧ', 'ﻨ', 'ﻥ', 'ﻦ', 'ڹ', 'ه', 'ﻪ', 'ﻫ', 'ﻩ', 'ﻬ', 'ہ', 'ھ', 'و', 'ﻭ', 'ﻮ', 'ۊ',
                                 'ؤ', 'ﺆ', 'ۆ', 'ۇ', 'ى', 'ﻰ', 'ﻯ', 'ي', 'ﻴ', 'ﻳ', 'ﻲ', 'ﻱ', 'ێ', 'ے', 'ی', 'ﯿ', 'ﯾ',
                                 'ﯽ', 'ﯼ', 'ئ', 'ﺋ', 'ﺌ', 'ە', 'ۀ', '۰', '؟', '+', '?', '!', '‍', 'Å', '{', '}', 'ږ',
                                 '【', '】', 'ﭗ', '〈', '〉', '﴾', '﴿', 'ۂ', 'ټ', 'ۺ', 'ݜ', 'ڕ', 'ګ', 'ݩ', 'ﺂ', 'ʍ', 'ﺙ',
                                 'ﯠ', 'ڰ', 'ﮪ', '۾', 'ݐ', 'ﮁ', 'ۍ', 'ݕ', '‼', 'ٹ', 'ݧ', '"']
        self.from_characters = ''.join(self._from_characters)
        self._to_characters = ['،', '(', ')', '\t', '\u200c', ' ', '0', '1', '1', '1', '2', '2', '2', '3', '3', '3',
                               '4',
                               '4', '4', '5', '5', '5', '6', '6', '6', '7', '7', '7', '8', '8', '8', '9', '9', '9', '.',
                               ':', '%', '/', '@', '#', '_', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f',
                               'g', 'g', 'h', 'h', 'i', 'i', 'j', 'j', 'k', 'k', 'l', 'l', 'm', 'm', 'n', 'n', 'o', 'o',
                               'p', 'p', 'q', 'q', 'r', 'r', 's', 's', 't', 't', 'u', 'u', 'v', 'v', 'w', 'w', 'x', 'x',
                               'y', 'y', 'z', 'z', 'ء', 'ء', 'ا', 'ا', 'ا', 'ا', 'ا', 'ا', 'ا', 'ا', 'ا', 'آ', 'آ', 'ب',
                               'ب', 'ب', 'ب', 'ب', 'پ', 'پ', 'پ', 'ت', 'ت', 'ت', 'ت', 'ت', 'ت', 'ت', 'ت', 'ث', 'ث', 'ث',
                               'ث', 'ث', 'ج', 'ج', 'ج', 'ج', 'ج', 'چ', 'چ', 'چ', 'چ', 'ح', 'ح', 'ح', 'ح', 'ح', 'خ', 'خ',
                               'خ', 'خ', 'خ', 'د', 'د', 'د', 'ذ', 'ذ', 'ذ', 'ر', 'ر', 'ر', 'ز', 'ز', 'ز', 'ژ', 'ژ', 'ژ',
                               'س', 'س', 'س', 'س', 'س', 'ش', 'ش', 'ش', 'ش', 'ش', 'ص', 'ص', 'ص', 'ص', 'ص', 'ض', 'ض', 'ض',
                               'ض', 'ض', 'ط', 'ط', 'ط', 'ط', 'ط', 'ظ', 'ظ', 'ظ', 'ظ', 'ع', 'ع', 'ع', 'ع', 'ع', 'غ', 'غ',
                               'غ', 'غ', 'غ', 'ف', 'ف', 'ف', 'ف', 'ف', 'ق', 'ق', 'ق', 'ق', 'ق', 'ک', 'ک', 'ک', 'ک', 'ک',
                               'ک', 'ک', 'ک', 'ک', 'ک', 'ک', 'گ', 'گ', 'گ', 'گ', 'گ', 'ل', 'ل', 'ل', 'ل', 'ﻻ', 'ﻻ', 'م',
                               'م', 'م', 'م', 'م', 'ن', 'ن', 'ن', 'ن', 'ن', 'ن', 'ه', 'ه', 'ه', 'ه', 'ه', 'ه', 'ه', 'و',
                               'و', 'و', 'و', 'و', 'و', 'و', 'و', 'ی', 'ی', 'ی', 'ی', 'ی', 'ی', 'ی', 'ی', 'ی', 'ی', 'ی',
                               'ی', 'ی', 'ی', 'ی', 'ئ', 'ئ', 'ئ', 'ه', 'ه', '.', '?', '+', '?', '!', '\u200c', 'A', '{',
                               '}', 'ر', '[', ']', 'پ', '<', '>', '(', ')', 'ه', 'ت', 'ش', 'ش', 'ر', 'ک', 'ن', 'ا', 'M',
                               'ث', 'و', 'گ', 'ه', 'م', 'پ', 'چ', 'ی', 'پ', '!', 'ت', 'ن', '"']
        self.to_characters = ''.join(self._to_characters)
        # endregion

    # تصحیح اشتباهاتی در متن مانند جداسازی حروف و ارقام با فاصله
    def organize_text(self, text: str):
        text = text.lower()

        result = regex.subf(r"([\u0600-\u06EF])([^\u0600-\u06EF_@\u200c#])", "{1} {2}", text)
        result = regex.subf(r"([^\u0600-\u06EF_@\u200c#])([\u0600-\u06EF])", "{1} {2}", result)
        result = regex.subf(r"([^\u06F0-\u06F9\n\t])([\u06F0-\u06F9])", "{1} {2}", result)
        result = regex.subf(r"([\u06F0-\u06F9])([^\u06F0-\u06F9\s\n\t])", "{1} {2}", result)

        return result

    # حذف آدرس های اینترنتی
    def replace_urls(self, text: str, replace_str=' URLT '):
        def matched(obj):
            return replace_str

        p1 = re.compile(
            r"([\n] |^)?((?:(?:https?|ftp)://)?(?:www\.)?(?:[\w\d-]+\.)+(?:\w+)(?:[/][\w\d_~:?#\@!%$&'()*+,;=`\[\]\.\-]+)*)([\n ]|$)?",
            re.UNICODE)
        text = text.lower()
        result = p1.sub(matched, text)
        return result

    # حذف آدرس های نامه های الکترونیکی
    def replace_emails(self, text: str, replace_str=' EMAILT '):
        def matched(obj):
            return replace_str

        text = text.lower()

        p1 = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', re.UNICODE)
        result = p1.sub(matched, text)
        return result

    # حذف نام های کاربری
    def replace_usernames(self, text: str, replace_str=' USERNAMET '):
        def matched(obj):
            return replace_str

        text = text.lower()

        p1 = re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', re.UNICODE)
        result = p1.sub(matched, text)
        return result

    # حذف هشتگ ها
    def replace_hashtags(self, text: str, replace_str=' HASHTAGT '):
        def matched(obj):
            return replace_str

        text = text.lower()

        p1 = re.compile(r'#(\w+)', re.UNICODE)
        l1_formatted_text = p1.sub(matched, text)
        return l1_formatted_text

    def replace_consecutive_spaces(self, text: str):
        def matched(obj):
            return ' '

        text = text.lower()

        p1 = re.compile(r'[ ]{2,}', re.UNICODE)
        result = p1.sub(matched, text)
        return result

    def replace_phone_numbers(self, text: str, replace_str=' PHONENUMBERT '):
        def matched(obj):
            return replace_str

        text = text.lower()

        p1 = re.compile(r'(\+[1-9][0-9]{11})|(0[0-9]{10})|([1-9][0-9]{7})', re.UNICODE)
        result = p1.sub(matched, text)
        return result

    def replace_emoji(self, text: str, replace_str=' EMOJIT '):
        def matched(obj):
            return replace_str

        text = text.lower()
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        result = emoji_pattern.sub(matched, text)
        return result

    def replace_duplicate_punctuation(self, text: str, replace_str=' . '):
        def matched(obj):
            return replace_str

        text = text.lower()
        punctuation_pattern = re.compile(r"[!\.\?⸮؟/,;:\\،؛]+", flags=re.UNICODE)
        result = punctuation_pattern.sub(matched, text)
        return result

    def edit_arabic_letters(self, text: str):
        # check string is empty
        if not text:
            return ""

        text = text.strip()
        text = text.translate({ord(x): y for (x, y) in zip(self.from_characters, self.to_characters)})
        return text

    def normalize_sentence(self, text: str):
        text = self.organize_text(text)
        text = self.replace_urls(text)
        text = self.replace_emails(text)
        text = self.replace_usernames(text)
        text = self.replace_hashtags(text)
        text = self.edit_arabic_letters(text)
        text = self.replace_phone_numbers(text)
        text = self.replace_emoji(text)
        text = self.replace_duplicate_punctuation(text)

        text = self.replace_consecutive_spaces(text)
        return text

    def normalize_sentence_trend_detection(self, text: str):
        text = self.replace_urls(text)
        text = self.replace_emails(text)
        text = self.replace_usernames(text)
        text = self.replace_hashtags(text)
        text = self.replace_phone_numbers(text)
        text = self.replace_emoji(text)

        return text


if __name__ == "__main__":
    # region tests
    sc = SentenceNormalizer()

    text = "2مرد رفتند به خانه"
    res = sc.organize_text(text)

    text = " علی گفت https://pypi.org/project/regex/ و بریم و دوباره گفت https://stackoverflow.com/questions/3031045/how-come-string-maketrans-does-not-work-in-python-3-1 و رفت."
    text = "علی رفت به https://youtu.be/isggeig9uhy"
    res = sc.replace_urls(text)

    text = "علی گفت a@b.c و رفت"
    res = sc.replace_usernames(text)
    res = sc.replace_emails(text)

    text = "علی گفت @ali_mm و رفت"
    res = sc.replace_usernames(text)
    res = sc.replace_emails(text)

    text = "علی گفت #iran44iran_22_k و #علی_گفتن و #علی_22 رفت"
    res = sc.replace_hashtags(text)

    text = "علی       گفت               و                            رفت"
    res = sc.replace_consecutive_spaces(text)

    text = "علی  ﻬمینﻙ گفت رﺐفت"
    res = sc.edit_arabic_letters(text)

    text = "علی گفت 09155555555 , +989155555555 و "
    res = sc.replace_phone_numbers(text)

    text = "علی گفت!!!! بیا بریم دیگه؟؟؟؟ بلابلا!؟!؟"
    res = sc.replace_duplicate_punctuation(text)
    # endregion
    print('Done!')
