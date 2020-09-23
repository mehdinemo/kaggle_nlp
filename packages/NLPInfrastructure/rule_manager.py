import codecs
import json

import requests

from NLPInfrastructure.resources import abbreviations, postWords
from NLPInfrastructure.resources import postWords
from NLPInfrastructure.resources import prepositions
from NLPInfrastructure.resources import stopWords
from NLPInfrastructure.resources import Months
from NLPInfrastructure.resources import Not1Gram


class RuleManager:
    def __init__(self):
        # load resources
        self.abbreviations = abbreviations
        self.abbreviations = [a.strip() for a in self.abbreviations]

        self.postWords = postWords
        self.postWords = [a.strip() for a in self.postWords]

        self.prepositions = prepositions
        self.prepositions = [a.strip() for a in self.prepositions]

        self.stopWords = stopWords
        self.stopWords = [a.strip() for a in self.stopWords]

        self.Months = Months
        self.Months = [a.strip() for a in self.Months]

        self.Not1Gram = Not1Gram
        self.Not1Gram = [a.strip() for a in self.Not1Gram]

        self.accept_threshold = 20
        self.decline_threshold = 5
        self.pos_tagger_url = "http://hazm.datamining.io/tagger?format=words"

    # region Public
    def apply_rules_red(self, ngrams):
        result = {}
        for ngram, freq in ngrams.items():
            if self.rule15(ngram, freq) & self.rule16(ngram, freq):
                result[ngram] = freq

        return result

    def apply_rules_yellow(self, ngrams):
        result = {}
        for ngram, freq in ngrams.items():
            if self.rule6(ngram) & \
                    self.rule7(ngram) & \
                    self.rule20(ngram) & \
                    self.rule22(ngram):
                result[ngram] = freq
        return result

    def apply_rules_blue(self, ngrams):
        result = {}
        # get poses
        poses = self._get_poses(ngrams)

        idx = 0
        for ngram, freq in ngrams.items():
            if self.rule1(ngram) & \
                    self.rule2(ngram) & \
                    self.rule5(ngram, poses[idx]) & \
                    self.rule10(ngram) & \
                    self.rule12(ngram, poses[idx]) & \
                    self.rule19(ngram) & \
                    self.rule24(ngram, poses[idx]) & \
                    self.rule26(ngram):
                result[ngram] = freq
            idx += 1
        return result

    # endregion

    # region Private

    """
    If ngram is deserved to be Removed(!) 
    rule* will return False
    """

    def rule1(self, ngram):
        parts = ngram.split(' ')
        if len(parts) == 2:
            if (parts[0] in self.stopWords):
                return False
            if (parts[1] in self.stopWords):
                return False
            if (parts[0] in self.prepositions):
                return False
            if (parts[1] in self.prepositions):
                return False


        elif len(parts) == 3:
            flag = 0
            for p in parts:
                if (p in self.stopWords):
                    flag += 1
                if (p in self.prepositions):
                    flag += 1
            if flag >= 2:
                return False
        return True

    def rule2(self, ngram):
        parts = ngram.split(' ')
        if parts[-1] in self.prepositions:
            return False
        return True

    def rule5(self, ngram, ngram_poses):
        parts = ngram.split(' ')

        ngram_poses = ngram_poses['words']
        if len(parts) != len(ngram_poses):
            return False

        if len(parts) == 2:
            if list(ngram_poses[1].values())[0] == "V":
                return False

        return True

    def rule6(self, ngram):
        parts = ngram.split(' ')
        if (parts[-1] == 'ی') | (parts[-1] == 'ي'):
            return False
        return True

    def rule7(self, ngram):
        parts = ngram.split(' ')
        if len(parts) == 2:
            if ('_' in parts[0]) | ('_' in parts[1]):
                return False

        elif len(parts) == 3:
            if ('_' in parts[0]) | ('_' in parts[1]) | ('_' in parts[2]):
                return False
        return True

    def rule10(self, ngram):
        parts = ngram.split(' ')
        if len(parts) == 2:
            if (str(parts[0]).isdigit()) & (not parts[1] in self.Months):
                return False
        return True

    def rule12(self, ngram, ngram_poses):
        parts = ngram.split(' ')

        ngram_poses = ngram_poses['words']
        if len(parts) != len(ngram_poses):
            return False

        if len(parts) == 3:
            if str(parts[0]).isdigit() & (list(ngram_poses[1].values())[0] == "N") & (
                    (parts[2] in self.stopWords) | (parts[2] in self.prepositions)):
                return False
        return True

    def rule15(self, ngram, freq):
        parts = ngram.split(' ')
        if (('و' in parts) | ('که' in parts) | ('یا' in parts)) & (freq < self.accept_threshold):
            return False
        return True

    def rule16(self, ngram, freq):
        parts = ngram.split(' ')
        if (('به' in parts) | ('برای' in parts) | ('تا' in parts) | ('از' in parts)) & (freq < self.decline_threshold):
            return False
        return True

    def rule19(self, ngram):
        parts = ngram.split(' ')
        if parts[0] in self.postWords:
            return False
        return True

    def rule20(self, ngram):
        parts = ngram.split(' ')
        if len(parts) == 1:
            if str(parts[0]).isdigit():
                return False
        return True

    def rule22(self, ngram):
        parts = ngram.split(' ')
        if len(parts) == 1:
            if len(parts[0]) == 1:
                return False

        if len(parts) == 2:
            if (len(parts[0]) == 1) | (len(parts[1]) == 1):
                return False

        if len(parts) == 3:
            if (len(parts[0]) == 1) | (len(parts[2]) == 1):
                return False

            if len(parts[1]) == 1:
                if not parts[1] in self.abbreviations:
                    return False

        return True

    def rule24(self, ngram, ngram_poses):
        parts = ngram.split(' ')

        ngram_poses = ngram_poses['words']
        if len(parts) != len(ngram_poses):
            return False

        if len(parts) == 1:
            for i in range(len(parts)):
                if list(ngram_poses[i].values())[0] == "V":
                    return False

        if len(parts) == 1:
            if (parts[0] in self.stopWords) | (parts[0] in self.prepositions) | \
                    (parts[0] in self.Not1Gram) | (parts[0] in self.abbreviations):
                return False
        return True

    def rule26(self, ngram):
        parts = ngram.split(' ')

        if len(parts) == 3:
            for p in parts:
                if (p in self.stopWords) | (parts[0] in self.prepositions):
                    for p in parts:
                        if str(p).isdigit():
                            return False

        return True

    def _get_poses(self, ngrams):
        body = []
        idx = 0
        for ngram, freq in ngrams.items():
            tmp = {"id": idx, "message": ngram}
            body.append(tmp)
            idx += 1

        headers = {'Content-type': 'application/json'}
        body = json.dumps(body, ensure_ascii=False, indent=4)
        resp = requests.post(self.pos_tagger_url, headers=headers, data=body.encode('utf-8'), timeout=None,
                             verify=False)
        dic = json.loads(resp.text)

        if not resp.ok:
            if 'error' in dic:
                return dic['error']
            else:
                return resp.status_code

        return dic

    def _get_pos(self, ngram):
        body = [{"id": 1, "message": ngram}]
        headers = {'Content-type': 'application/json'}
        resp = requests.post(self.pos_tagger_url, headers=headers, data=json.dumps(body).encode('utf-8'))
        dic = json.loads(resp.text)

        if not resp.ok:
            if 'error' in dic:
                return dic['error']
            else:
                return resp.status_code

        return dic

    # endregion


if __name__ == "__main__":
    rlm = RuleManager()

    ngrams = {
        '11 ساله هستم': 20,
        'علی به مدرسه': 3,
        'میرود': 50,
        'می رود': 5
    }
    print(rlm.apply_rules_blue(ngrams))
    print(rlm.apply_rules_red(ngrams))
    print(rlm.apply_rules_yellow(ngrams))
