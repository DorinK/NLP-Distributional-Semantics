from DistributionalSimilarities import SPECIAL_FUNCTION_WORDS, FUNCTION_WORDS_TAGS


def is_function_word(word):
    lemma, pos_tag = word.split('\t')[2], word.split('\t')[3]
    return True if lemma in SPECIAL_FUNCTION_WORDS or pos_tag in FUNCTION_WORDS_TAGS else False


class ContextWordSentence(object):

    def __init__(self):
        pass

    def get_contexts(self, sentence):

        # Split and filter function words.
        lemmas = [current_sentence.split('\t')[2] for current_sentence in sentence if
                  not is_function_word(current_sentence)]

        # Get the contexts for each word (which is not function word) in the sentence.
        contexts = [lemmas[:i] + lemmas[i + 1:] for i in range(len(lemmas))]

        return lemmas, contexts


class ContextWordWindow(object):

    def __init__(self, window_size=2):
        self.window_size = window_size

    def get_contexts(self, sentence):

        # Split and filter function words.
        lemmas = [current_sentence.split('\t')[2] for current_sentence in sentence if
                  not is_function_word(current_sentence)]

        # Get the contexts for each word (which is not function word) in the sentence.
        contexts = [lemmas[max(0, i - self.window_size):i]
                    + lemmas[i + 1: min(len(lemmas), i + self.window_size + 1)] for i in range(len(lemmas))]

        return lemmas, contexts


class ContextWordDependency(object):

    def __init__(self):
        pass

    def get_contexts(self, sentence):

        # Preprocessing.
        words = [word.split('\t') for word in sentence]
        contexts = [[] for _ in range(len(sentence))]
        lemmas = [word[2] for word in words]

        for i in range(0, len(sentence)):

            dependency_id = int(words[i][6]) - 1  # head.
            dependency_label = words[i][7]  # Dependency relation to the HEAD.

            if dependency_id != -1:

                # If the dependency word and the current word are both not function words.
                if words[dependency_id][3] != 'IN' and not is_function_word(
                        sentence[dependency_id]) and not is_function_word(sentence[i]):

                    # Add these features.
                    contexts[i].append(str((lemmas[dependency_id], dependency_label, '↑')))
                    contexts[dependency_id].append(str((lemmas[i], dependency_label, '↓')))

                # If the dependency word is a preposition.
                elif words[dependency_id][3] == 'IN' and not is_function_word(sentence[i]):

                    # Get the preposition dependency.
                    prep_dependency_id = int(words[dependency_id][6]) - 1
                    prep_dependency_label = words[dependency_id][7]

                    # Continue only if the word connected to the preposition is not a function word.
                    if prep_dependency_id != -1 and not is_function_word(sentence[prep_dependency_id]):

                        # Extract the connected word.
                        prep_dependency_connected_word = words[prep_dependency_id][2]

                        contexts[i].append(str((prep_dependency_connected_word, prep_dependency_label, "↑",
                                                lemmas[dependency_id])))  # case ii of preposition

                        contexts[prep_dependency_id].append(  # case i of preposition
                            str((lemmas[i], prep_dependency_label, "↓", lemmas[dependency_id])))

        return lemmas, contexts
