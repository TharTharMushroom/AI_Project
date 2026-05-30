from typing import List


class TrigramLanguageModel:

    def __init__(self,
                 trigram_counts,
                 bigram_counts,
                 context_counts,
                 prev_word_counts,
                 unigram_counts):

        self.trigram_counts = trigram_counts
        self.bigram_counts = bigram_counts

        self.context_counts = context_counts
        self.prev_word_counts = prev_word_counts

        self.unigram_counts = unigram_counts
        self.total_unigram_count = sum(unigram_counts.values())

    def get_vocabulary(self):
        return self.unigram_counts.keys()

    def unigram_prob(self, word):

        if word not in self.unigram_counts:
            return 0

        return self.unigram_counts[word] / self.total_unigram_count

    def bigram_prob(self, prev_word, word):

        if prev_word not in self.bigram_counts:
            return 0

        count = self.bigram_counts[prev_word].get(word, 0)

        return count / self.prev_word_counts[prev_word]

    def trigram_prob(self, word1, word2, word3):

        context = (word1, word2)

        if context not in self.trigram_counts:
            return 0

        count = self.trigram_counts[context].get(word3, 0)

        return count / self.context_counts[context]

    def get_probability(self, word1, word2, word3):

        trigram = self.trigram_prob(word1, word2, word3)
        bigram = self.bigram_prob(word2, word3)
        unigram = self.unigram_prob(word3)

        return (
            0.7 * trigram +
            0.2 * bigram +
            0.1 * unigram
        )


def estimate_trigram_lm(train_seqs):

    trigram_counts = {}

    bigram_counts = {}

    context_counts = {}

    prev_word_counts = {}

    unigram_counts = {}

    for seq in train_seqs:

        for i in range(2, len(seq)):

            w1 = seq[i - 2]
            w2 = seq[i - 1]
            w3 = seq[i]

            context = (w1, w2)

            if context not in trigram_counts:
                trigram_counts[context] = {}

            trigram_counts[context][w3] = (
                trigram_counts[context].get(w3, 0) + 1
            )

            context_counts[context] = (
                context_counts.get(context, 0) + 1
            )

            if w2 not in bigram_counts:
                bigram_counts[w2] = {}

            bigram_counts[w2][w3] = (
                bigram_counts[w2].get(w3, 0) + 1
            )

            prev_word_counts[w2] = (
                prev_word_counts.get(w2, 0) + 1
            )

            unigram_counts[w3] = (
                unigram_counts.get(w3, 0) + 1
            )

    return TrigramLanguageModel(
        trigram_counts,
        bigram_counts,
        context_counts,
        prev_word_counts,
        unigram_counts
    )