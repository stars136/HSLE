import pickle
from math import log10

from pagerank_weighted import pagerank_weighted_scipy as _pagerank
from preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from commons import build_graph as _build_graph
from commons import remove_unreachable_nodes as _remove_unreachable_nodes


def _set_graph_edge_weights(graph):
    for sentence_1 in graph.nodes():
        for sentence_2 in graph.nodes():

            edge = (sentence_1, sentence_2)
            if sentence_1 != sentence_2 and not graph.has_edge(edge):
                similarity = _get_similarity(sentence_1, sentence_2)
                if similarity != 0:
                    graph.add_edge(edge, similarity)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)


def _create_valid_graph(graph):
    nodes = graph.nodes()

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


def _get_similarity(s1, s2):
    words_sentence_one = s1.split()
    words_sentence_two = s2.split()

    common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one))
    log_s2 = log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0

    return common_word_count / (log_s1 + log_s2)


def _count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))


def _format_results(extracted_sentences, split, score):
    if score:
        return [(sentence.text, sentence.score) for sentence in extracted_sentences]
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _add_scores_to_sentences(sentences, scores):
    for sentence in sentences:
        # Adds the score to the object if it has one.
        if sentence.token in scores:
            sentence.score = scores[sentence.token]
        else:
            sentence.score = 0
        #print(sentence.token, sentence.score)
        #如何匹配词典则直接加分
        #sentence 是一个列表元组，Original unit即原始的单元，Processed unit则是去除停用词，去掉单词后缀形态等清洗手段之后剩下的内容


def _get_sentences_with_word_count(sentences, words):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided.
    """
    word_count = 0
    selected_sentences = []
    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(words - word_count - words_in_sentence) > abs(words - word_count):
            return selected_sentences

        selected_sentences.append(sentence)
        word_count += words_in_sentence

    return selected_sentences


def _extract_most_important_sentences(sentences, ratio, words):
    sentences.sort(key=lambda s: s.score, reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio.
    if words is None:
        length = len(sentences) * ratio
        return sentences[:int(length)]

    # Else, the ratio is ignored.
    else:
        return _get_sentences_with_word_count(sentences, words)


def summarize(text, ratio=0.2, cid=None, words=None, language="english", split=False, scores=False, additional_stopwords=None):
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text, language, additional_stopwords)
    #print(sentences[0], sentences[0].token)  #list,token为清洗，分词后的句子

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return [] if split else ""

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = _pagerank(graph)
    if cid is not None:
        with open('dicword.pkl', 'rb') as f:
            dic = pickle.load(f)
        corew = set([])
        for c in cid:
            corew.intersection(dic[c])
        for key in pagerank_scores.keys():
            temp = set(key.split(' '))
            #print(len(temp.intersection(corew))/len(temp))
            pagerank_scores[key] += len(temp.intersection(corew))/len(temp)
            #if len(temp.intersection(corew)) > 0:
            
    #这里考虑改进，分数加入关键词加成

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences, pagerank_scores)

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_most_important_sentences(sentences, ratio, words)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split, scores)


def get_graph(text, language="english"):
    sentences = _clean_text_by_sentences(text, language)

    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    return graph

if __name__ == "__main__":
    text = "With the development of service computing, the increasing number and diversity of web services make it an intractable task to find suitable services. \
        Service composition, service selection and recommendation have become the focus of service computing. \
        As the basis of service network research, service link prediction can explore the composition mode between services, and facilitate the study of service selection, service composition and service recommendation. \
        However, the existing link prediction methods are mainly based on artificial modeling and derivation, which cannot make full use of the global structure information and perform poorly in complex networks. \
        A challenging problem in service link prediction is the heterogeneous and sparse characteristics of the service network. \
        Therefore, we propose a novel web service link prediction method based on heterogeneous graph attention network. \
        By analyzing the interaction between services, five types of neighbors that are associated with service links are chosen, and two levels of attention are used to learn their importance and calculate the embedding of service links. \
        In addition, in order to improve efficiency, we design a Service-TextRank algorithm to extract the key information of the service description. \
        Extensive experimental results on real world data-ProgrammableWeb validate the effectiveness of our approach."
    print(summarize(text=text, words=50, cid=[22481,22482]))