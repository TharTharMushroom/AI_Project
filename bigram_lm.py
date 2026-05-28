from typing import List


class BigramLanguageModel(object):
    """
    Class that implements a bigram language model

    Attributes:
        bigram_counts: A List of dictionaries. The ith dictionary contains counts for words following word i. So if
        i = 67 corresponds to "of", bigram_counts[67] stores a dict of counts like {the: 125, my: 23. ...} for all words
        that we ever see after "of"
        prev_word_counts: A List of counts for each word appearing as a "previous" word, or "context" word.
        unigram_counts: A List of counts for words appearing as the "current" word. These are the same counts as those
        estimated by the UnigramLanguageModel
    """
    def __init__(self, bigram_counts: dict, prev_word_counts: dict, unigram_counts: dict):
        self.bigram_counts = bigram_counts
        self.prev_word_counts = prev_word_counts
        self.unigram_counts = unigram_counts
        self.total_unigram_count = sum([unigram_counts[word] for word in unigram_counts.keys()])
        self.lam = 0.9
        self.use_multiplicative = True

    def get_vocabulary(self):
        """
        :return: A set containing the vocabulary of the
        """
        return self.unigram_counts.keys()

    def _get_unigram_probability(self, word: str) -> float:
        """
        Helper method to calculate the unigram probability of the given word
        :param word: The index of the word to get the unigram probability for
        :return: The unigram probability of the word
        """
        return float(self.unigram_counts[word])/self.total_unigram_count

    def get_probability(self, prev_word: str, word: str) -> float:
        """
        Computes the probability P(word | prev_word)
        :param prev_word: the previous word
        :param word: the next word (candidate) to score
        :return: The float bigram probability of word given prev_word
        """
        counts_after_prev_word = self.bigram_counts[prev_word]
        if word in counts_after_prev_word:
            next_word_in_context_count = counts_after_prev_word[word]
        else:
            next_word_in_context_count = 0
        return self.lam * next_word_in_context_count / self.prev_word_counts[prev_word] + (
                    1 - self.lam) * self._get_unigram_probability(word)


def estimate_bigram_lm(train_seqs: List[List[str]]) -> BigramLanguageModel:
    bigram_counts = {}
    # The following two have to be different because of the start/end of sequence characters
    unigram_counts = {}
    prev_word_counts = {}
    for train_seq in train_seqs:
        for i in range(1, len(train_seq)):
            prev_word = train_seq[i-1]
            word = train_seq[i]
            if prev_word not in bigram_counts:
                bigram_counts[prev_word] = {}
            if word in bigram_counts[prev_word]:
                bigram_counts[prev_word][word] += 1
            else:
                bigram_counts[prev_word][word] = 1
            if prev_word not in prev_word_counts:
                prev_word_counts[prev_word] = 1
            else:
                prev_word_counts[prev_word] += 1
            if word not in unigram_counts:
                unigram_counts[word] = 1
            else:
                unigram_counts[word] += 1
    return BigramLanguageModel(bigram_counts, prev_word_counts, unigram_counts)
