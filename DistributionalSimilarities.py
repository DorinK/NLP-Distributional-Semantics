import gc
from tabulate import tabulate
from CoOccurrenceTypes import *
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import ast
import time

K = 20
TOP_50 = 50
LEMMA_MIN_OCCURRENCES = 100
CONTEXT_MIN_OCCURRENCES = 75
CONTEXTS_PER_WORD_THRESHOLD = 100

TARGET_WORDS = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]

FUNCTION_WORDS_TAGS = {",", ".", "(", ")", "``", "''", ":", "$", 'IN', 'PRP', 'PRP$', 'WP', 'WP$', 'DT', 'WDT', 'CC',
                       'CD', 'PDT', 'Particle', 'UH', 'TO', 'EX', 'LS', 'MD', 'POS'}

SPECIAL_FUNCTION_WORDS = {'while', 'therefore', 'off', 'they', 'their', 'my', 'do', 'his', 'without', 'and', 'with',
                          'to', 'this', 'he', 'either', 'neither', 'her', 'are', "'s", 'that', 'them', 'in', 'more',
                          'after', 'when', 'or', 'much', 'not', 'but', 'one', 'could', 'the', 'is', 'by', 'via', 'for',
                          'at', 'have', '[', 'an', ']', 'can', 'she', 'a', 'as', 'where', 'only', 'be', 'which', 'it',
                          'between', 'am', 'has', 'him', 'both', 'got', 'who', 'anybody', 'also', 'of', 'on', 'from'}


class DistributionalSimilarities:

    def __init__(self, file_name):

        self.sentences = list()
        self.lemma_count = Counter()
        self.common_lemmas = Counter()

        # Preprocess the data.
        self.read_file(file_name)
        self.get_lemma_count()
        self.filter_lemmas()

        # Produce a file with 50 most common content words.
        # self.write_most_common_words_to_file()

        # Mappings - assigning a unique index to each word.
        self.i2w = np.array([w for w in self.common_lemmas])
        self.w2i = {w: i for i, w in enumerate(self.common_lemmas)}

        self.first_order_similarities = defaultdict(list)
        self.second_order_similarities = defaultdict(list)

        self.co_occurrence_types = {'Sentence Co-occurrence': ContextWordSentence(),
                                    'Window Co-occurrence': ContextWordWindow(),
                                    'Dependency Co-occurrence': ContextWordDependency()}

        self.content_words_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.word_to_attributes_matrix = []
        self.attribute_to_words_matrix = defaultdict(list)

    def read_file(self, file_name):

        sentence = []

        # Read the corpus file.
        with open(file_name, "r", encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:

            if line != '\n':  # If the current sentence not ended yet.
                sentence.append(line.strip())

            else:  # The current sentence has ended.
                self.sentences.append(sentence)
                sentence = []

    def get_lemma_count(self):

        for sentence in self.sentences:

            for word in sentence:

                # Split the line into the word's attributes.
                word_attributes = word.split('\t')
                pos_tag = word_attributes[3]
                lemma = word_attributes[2]

                # Count only content words appearances (in the lemma form).
                if pos_tag not in FUNCTION_WORDS_TAGS:
                    self.lemma_count[lemma] += 1

    def filter_lemmas(self):

        # Initialize a new counter with the values of the current counter.
        self.common_lemmas = Counter(self.lemma_count.elements())

        # Remove special function words and words that do not exceed the minimum occurrence threshold.
        for lemma, count in self.lemma_count.items():
            if lemma in SPECIAL_FUNCTION_WORDS or count < LEMMA_MIN_OCCURRENCES:
                del self.common_lemmas[lemma]

    def write_most_common_words_to_file(self):

        # Produce a file with 50 most common content words.
        with open("counts_words.txt", "w") as file:
            for i, (lemma, count) in enumerate(self.common_lemmas.most_common(TOP_50), 1):
                file.write("{} {}{}".format(lemma, count, '\n' if i != TOP_50 else ""))

    def get_content_words_counts(self, co_occurrence_type):

        self.content_words_counts = defaultdict(Counter)

        for sentence in self.sentences:

            # Extract the words and the contexts.
            words, contexts = co_occurrence_type.get_contexts(sentence)

            for (word, context) in zip(words, contexts):

                if word not in self.common_lemmas:
                    continue  # Filter rare words.

                context_counts_for_word = self.content_words_counts[word]
                for context_word in context:
                    context_counts_for_word[context_word] += 1

    def filter_features(self, is_dependency):

        # For each word and its features.
        for word, contexts in self.content_words_counts.items():

            # Use a second counter in order to perform the filtering.
            contexts_cpy = Counter(self.content_words_counts[word])

            for context in contexts:  # For each feature

                if is_dependency:  # If it's the co-occurrence type of dependency.

                    # Represent the feature as a tuple.
                    lemma = ast.literal_eval(context)[0]

                    # Remove features that do not exceed the minimum occurrence threshold.
                    if self.lemma_count[lemma] < CONTEXT_MIN_OCCURRENCES:
                        del contexts_cpy[context]

                # Remove features that do not exceed the minimum occurrence threshold.
                elif self.lemma_count[context] < CONTEXT_MIN_OCCURRENCES:
                    del contexts_cpy[context]

            # Keep only the 100 most common contexts per word.
            self.content_words_counts[word] = Counter(OrderedDict(contexts_cpy.most_common(CONTEXTS_PER_WORD_THRESHOLD)))

    def get_context_counts(self):

        self.context_counts = Counter()

        # Count how many times each word appears as a context of another word.
        for content_word, context in self.content_words_counts.items():
            for context_word in context:
                self.context_counts[context_word] += self.content_words_counts[content_word][context_word]

    def write_top_dependency_contexts_to_file(self):

        # Produce a file with the counts of the 50 top dependency contexts.
        with open("counts_contexts_dep.txt", "w", encoding='utf-8') as file:
            for i, (feature, count) in enumerate(self.context_counts.most_common(TOP_50), 1):
                file.write("{} {}{}".format(feature, count, '\n' if i != TOP_50 else ""))

    def compute_PMI(self, x, y):

        # The probability that the word-attribute co-occurrence will have x as the word.
        p_x = sum(self.content_words_counts[x].values()) / self.total_num_pairs

        # The probability that the word-attribute co-occurrence will have y as the attribute.
        p_y = self.context_counts[y] / self.total_num_pairs

        # The probability that the word-attribute co-occurrence will have x as the word and y as the attribute.
        p_x_y = (self.content_words_counts[x][y]) / self.total_num_pairs

        # Return the PMI score according to the PPMI concept - turn negative PMI scores into 0.
        return max(np.log(p_x_y / (p_x * p_y)), 0)

    def get_word_to_attributes_matrix(self):

        # The columns matrix - each column is a word.
        self.word_to_attributes_matrix = [None] * len(self.common_lemmas)

        # Compute the normalization factor.
        self.total_num_pairs = sum([sum(self.content_words_counts[word].values()) for word in self.content_words_counts])

        # Compute a features vector to each word.
        for word in self.common_lemmas:

            attributes_vec = {}

            # For each word-attribute pair compute a PMI score.
            for att in self.content_words_counts[word].keys():

                PMI = self.compute_PMI(word, att)

                if PMI > 0:  # Ignore PMI scores of 0.
                    attributes_vec[att] = PMI

            self.word_to_attributes_matrix[self.w2i[word]] = attributes_vec

    def get_attribute_to_words_matrix(self):

        # The rows matrix - each row is an attribute.
        self.attribute_to_words_matrix = defaultdict(list)

        for i, attributes in enumerate(self.word_to_attributes_matrix):
            for att in attributes:  # For each attribute, add the PMI score computed with the current word to the list.
                self.attribute_to_words_matrix[att].append((i, attributes[att]))

    def cosine_similarity(self, word_index):

        ### Compute the numerator of the Cosine ###

        # Extract the features vector of the target word.
        word_attributes = self.word_to_attributes_matrix[word_index]

        # Initialize the similarity results vector.
        similarity_results = [0] * len(self.common_lemmas)

        # The efficient algorithm.
        for att in word_attributes:
            for v in self.attribute_to_words_matrix[att]:
                similarity_results[v[0]] += word_attributes[att] * self.word_to_attributes_matrix[v[0]][att]

        ### Compute the denominator of the Cosine ###

        # Sum the u squares.
        sum_u_att_squares = sum([word_attributes[att] * word_attributes[att] for att in word_attributes])

        for i in range(len(self.common_lemmas)):

            # Sum the v squares.
            v_attributes = self.word_to_attributes_matrix[i]
            sum_v_att_squares = sum([v_attributes[att] * v_attributes[att] for att in v_attributes])

            # Avoid division by 0.
            if sum_u_att_squares == 0 or sum_v_att_squares == 0:
                similarity_results[i] = 0

            else:  # Divide the numerator by the denominator and get the similarity score to the i'th word.
                similarity_results[i] /= np.sqrt(sum_u_att_squares * sum_v_att_squares)

        return similarity_results

    def calculate_similarities_of_co_occurrence(self, key, co_occurrence_type):

        self.get_content_words_counts(co_occurrence_type)
        self.filter_features(isinstance(co_occurrence_type, ContextWordDependency))
        self.get_context_counts()

        # Produce a file with the counts of the 50 top dependency contexts.
        # if isinstance(co_occurrence_type, ContextWordDependency):
        #     self.write_top_dependency_contexts_to_file()

        # Compute the columns matrix.
        self.get_word_to_attributes_matrix()

        # Delete unnecessary resources.
        del self.context_counts
        gc.collect()

        # Compute the rows matrix.
        self.get_attribute_to_words_matrix()

        for word in TARGET_WORDS:

            # Extract 1st-order similarities - attributes with highest PMI values in the target word’s vector.
            context_attributes = Counter(self.word_to_attributes_matrix[self.w2i[word]]).most_common(K)
            top_context_attributes = [context[0] for context in context_attributes]
            self.first_order_similarities[key].append(top_context_attributes)

            # Compute 2nd-order similarities using the Cosine similarity measure.
            similar_words = np.array(self.cosine_similarity(self.w2i[word]))
            most_similar = similar_words.argsort()[-2:-K - 2:-1]
            self.second_order_similarities[key].append(list(self.i2w[most_similar]))

        # Delete unnecessary resources.
        del self.content_words_counts, self.word_to_attributes_matrix, self.attribute_to_words_matrix, similar_words,\
            most_similar, context_attributes, top_context_attributes
        gc.collect()

    def print_similarities(self):

        # Print similarities tables for each similarity order.
        for i, order in enumerate([self.first_order_similarities, self.second_order_similarities]):

            print()
            print("## {} Order Similarity ##".format('1st' if i + 1 == 1 else '2nd'))

            # Print the similarities table for the current target word.
            for j, word in enumerate(TARGET_WORDS):
                print()
                print(" +-- " + word + " --+")
                types = list(self.co_occurrence_types.keys())
                table_data = [[sent_item, window_item, dep_item] for sent_item, window_item, dep_item in
                              zip(order[types[0]][j], order[types[1]][j], order[types[2]][j])]
                print(tabulate(table_data, headers=types, tablefmt='grid', colalign=("center", "center", "center")))

            print()


if __name__ == "__main__":

    start_time = time.time()

    # Initialize a new object.
    DS = DistributionalSimilarities("wikipedia.sample.trees.lemmatized")

    # Calculate the similarities foe each co-occurrence type.
    for key, co_occurrence in DS.co_occurrence_types.items():
        DS.calculate_similarities_of_co_occurrence(key, co_occurrence)

    # Print the similarity tables.
    DS.print_similarities()

    print("Finished after {} minutes\n".format((time.time() - start_time) / 60))
