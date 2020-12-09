"""General utilities for training.

Author:
    Shrey Desai
"""

import os
import json
import gzip
import pickle

import torch
from tqdm import tqdm

import spacy
from nltk.stem.lancaster import LancasterStemmer

def cuda(args, tensor):
    """
    Places tensor on CUDA device (by default, uses cuda:0).

    Args:
        tensor: PyTorch tensor.

    Returns:
        Tensor on CUDA device.
    """
    if args.use_gpu and torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def unpack(tensor):
    """
    Unpacks tensor into Python list.

    Args:
        tensor: PyTorch tensor.

    Returns:
        Python list with tensor contents.
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy().tolist()


def load_dataset(path):
    """
    Loads MRQA-formatted dataset from path.

    Args:
        path: Dataset path, e.g. "datasets/squad_train.jsonl.gz"

    Returns:
        Dataset metadata and samples.
    """
    with gzip.open(path, 'rb') as f:
        elems = [
            json.loads(l.rstrip())
            for l in tqdm(f, desc=f'loading \'{path}\'', leave=False)
        ]
    meta, samples = elems[0], elems[1:]
    return (meta, samples)


def load_cached_embeddings(path):
    """
    Loads embedding from pickle cache, if it exists, otherwise embeddings
    are loaded into memory and cached for future accesses.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    bare_path = os.path.splitext(path)[0]
    cached_path = f'{bare_path}.pkl'
    if os.path.exists(cached_path):
        return pickle.load(open(cached_path, 'rb'))
    embedding_map = load_embeddings(path)
    pickle.dump(embedding_map, open(cached_path, 'wb'))
    return embedding_map


def load_embeddings(path):
    """
    Loads GloVe-style embeddings into memory. This is *extremely slow* if used
    standalone -- `load_cached_embeddings` is almost always preferable.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    embedding_map = {}
    with open(path) as f:
        next(f)  # Skip header.
        for line in f:
            try:
                pieces = line.rstrip().split()
                embedding_map[pieces[0]] = [float(weight) for weight in pieces[1:]]
            except:
                pass
    return embedding_map


def search_span_endpoints(start_probs, end_probs, passage, question, window=15):
    """
    Finds an optimal answer span given start and end probabilities.
    Specifically, this algorithm finds the optimal start probability p_s, then
    searches for the end probability p_e such that p_s * p_e (joint probability
    of the answer span) is maximized. Finally, the search is locally constrained
    to tokens lying `window` away from the optimal starting point.

    Args:
        start_probs: Distribution over start positions.
        end_probs: Distribution over end positions.
        window: Specifies a context sizefrom which the optimal span endpoint
            is chosen from. This hyperparameter follows directly from the
            DrQA paper (https://arxiv.org/abs/1704.00051).

    Returns:
        Optimal starting and ending indices for the answer span. Note that the
        chosen end index is *inclusive*.
    """
    st = LancasterStemmer()
    nlp = spacy.load('en_core_web_sm')
    list_spans = list()

    doc_context = nlp(' '.join(passage))

    if 'name' in question or 'Name' in question:

        start = 0
        for sent in doc_context.sents:
            doc = nlp(str(sent))

            end = start + len(str(sent).split(' '))

            if len(doc.ents) > 0:
                list_spans.append((start,end))

            start = end



    # doc_question = nlp(' '.join(question))

    # question_name_ents = [str(ent) for ent in doc_question.ents]

    # roots_question = list()

    # for sent in doc_question.sents:
    #     roots_question = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]

    # start = 0

    # for sent in doc_context.sents:

    #     context_name_ents = [str(ent) for ent in sent.ents]

    #     roots_context = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]

    #     end = start + len(str(sent).split(' '))

    #     added = False

    #     for root in roots_question:
    #         if root in roots_context:
    #             list_spans.append((start, end))
    #             added = True
    #             break
    #         if added:
    #             break

    #     if added:
    #         for i in question_name_ents:
    #             for j in context_name_ents:
    #                 if i in j or j in i:
    #                     list_spans.append((start, end))
    #                     added = True
    #                     break
    #             if added:
    #                 break

    # max_start_index = -1
    # max_end_index = -1

    # for span in list_spans:
    #     max_start_index = start_probs.index(max(start_probs[span[0]: span[1]]))
    #     max_end_index = -1
    #     max_joint_prob = 0.

    #     for end_index in range(max_start_index, span[1]):
    #         if max_start_index <= end_index <= max_start_index + window:
    #             joint_prob = start_probs[max_start_index] * end_probs[end_index]
    #             if joint_prob > max_joint_prob:
    #                 max_joint_prob = joint_prob
    #                 max_end_index = end_index




    # max_start_index = start_probs.index(max(start_probs))
    # max_end_index = -1
    # max_joint_prob = 0.

    # for end_index in range(len(end_probs)):
    #     if max_start_index <= end_index <= max_start_index + window:
    #         joint_prob = start_probs[max_start_index] * end_probs[end_index]
    #         if joint_prob > max_joint_prob:
    #             max_joint_prob = joint_prob
    #             max_end_index = end_index

    max_start_index = -1
    max_end_index = -1
    max_joint_prob = 0

    for span in list_spans:
        max_start, max_end = span

        for i in range(max_start, max_end):
            for j in range(max_start + 1, max_end):
                joint_prob = start_probs[i] * end_probs[j]
                if joint_prob > max_joint_prob:
                    max_start_index = i
                    max_end_index = j
                    max_joint_prob = joint_prob

    return (max_start_index, max_end_index)
