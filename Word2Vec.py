import time
import pandas as pd
import numpy as np
from tabulate import tabulate
from collections import defaultdict
import sklearn.preprocessing as preprocessing

K = 20

BOW = 'bow5'
DEPENDENCY = 'deps'
VEC_VERSION = {'Bag of Words': BOW, 'Dependency-Based': DEPENDENCY}

TARGET_WORDS = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]


def preprocess(file_name):

    # Read the vectors into a numpy array.
    data = pd.read_csv(file_name, sep=' ', header=None).values

    # Create mappings.
    items = np.array([row[0] for row in data])
    i2i = {item: i for i, item in enumerate(items)}

    # Normalize the vectors.
    m = np.array([row[1:].astype(np.float32) for row in data])
    M = preprocessing.normalize(m, norm='l2')

    return items, i2i, M


def get_top_K(word_vec, M, k2i, bias):

    # Compute the dot product of the target word vector with the vectors in the matrix.
    sims = M.dot(word_vec)

    # Extract the top K words / contexts.
    most_similar_ids = sims.argsort()[-bias:-K - bias:-1]
    sim_words = k2i[most_similar_ids]

    return sim_words


def print_similarities():

    # Print similarities tables for each similarity order.
    for i, order in enumerate([first_order, second_order]):

        print()
        print("## {} Order Similarity ##".format('1st' if i + 1 == 1 else '2nd'))

        # Print the similarities table for the current target word.
        for j, word in enumerate(TARGET_WORDS):
            print()
            print(" +-- " + word + " --+")
            table_data = [[bow_item, dep_item] for bow_item, dep_item in zip(order[BOW][j], order[DEPENDENCY][j])]
            print(tabulate(table_data, headers=VEC_VERSION.keys(), tablefmt='grid', colalign=("center", "center")))

        print()


if __name__ == '__main__':

    start_time = time.time()
    first_order, second_order = defaultdict(list), defaultdict(list)

    for vec_version in VEC_VERSION.values():

        # Preprocess the data.
        words, w2i, W = preprocess(vec_version + '.words')
        contexts, c2i, C = preprocess(vec_version + '.contexts')

        for word in TARGET_WORDS:

            # Extract the target word's vector.
            word_vec = W[w2i[word]]

            # Compute 1st-order and 2nd order similarities.
            first_order[vec_version].append(list(get_top_K(word_vec, C, contexts, 1)))
            second_order[vec_version].append(list(get_top_K(word_vec, W, words, 2)))

    print_similarities()
    print("Finished after {} minutes.".format((time.time() - start_time) / 60))
