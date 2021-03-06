"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch

from torch.utils.data import Dataset
from random import shuffle
from utils import cuda, load_dataset
from torch.utils.data import Dataset, DataLoader
# import transformers
import spacy
from nltk.stem.lancaster import LancasterStemmer

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """
    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (_, passage, question, _, _, _, _) in samples:
            for token in itertools.chain(passage, question):
                vocab[token.lower()] += 1
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words
    
    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]


class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """
    def __init__(self, args, path):
        self.args = args
        self.meta, self.elems = load_dataset(path)
        self.samples = self._create_samples()
        self.tokenizer = None
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizer.pad_token_id \
            if self.tokenizer is not None else 0

    def _create_samples(self):
        """_create_samples
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """

        st = LancasterStemmer()
        nlp = spacy.load('en_core_web_sm')

        samples = []

        count = 0

        for elem in self.elems:

            if count % 1000 == 0:
                print(count)

            # Unpack the context paragraph. Shorten to max sequence length.

            # print('Elem: ', elem['context'])
            # print('\n')

            # print(elem)

            passage = [
                token.lower() for (token, offset) in elem['context_tokens']
            ][:self.args.max_context_length]

            passage_not_lower = [
                token for (token, offset) in elem['context_tokens']
            ][:self.args.max_context_length]

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            # print(elem['context'])

            for qa in elem['qas']:

                # print('Passage: ', passage)

                non_tokenized_context = ' '.join(passage_not_lower)

                question = [
                    token.lower() for (token, offset) in qa['question_tokens']
                ][:self.args.max_question_length]

                if 'who' in question:
                    doc_context = nlp(non_tokenized_context)

                    temp = ''

                    for sent in doc_context.sents:
                        doc = nlp(str(sent))

                        name_ents = [str(ent.label_) for ent in doc.ents]

                        # if 'EVENT' in name_ents or 'DATE' in name_ents or 'TIME' in name_ents:
                        #     temp += str(sent) + ' '
                        #     added = True

                        # if 'PERSON' in name_ents or 'ORG' in name_ents:
                        #     temp += str(sent) + ' '
                        #     added = True

                        added = False

                        if len(doc.ents) > 0:
                            temp += str(sent) + ' '
                            added = True

                        if not added:
                            tmp = str(sent)
                            tmp = tmp.split(' ')
                            tmp = [PAD_TOKEN] * len(tmp)
                            tmp = ' '.join(tmp)
                            temp += tmp + ' '

                    temp = temp[0: -1]

                    passage = [chunk.lower() for chunk in temp][:self.args.max_context_length]

                



                # # print('NT: ', non_tokenized_context)
                # # print(non_tokenized_context.split(' '))
                # non_tokenized_question = qa['question']

                # # roots_question = []
                # # roots_context = []

                # doc_context = nlp(non_tokenized_context)
                # doc_question = nlp(non_tokenized_question)

                # # question_name_ents = [str(ent) for ent in doc_question.ents]

                # temp = ''

                # for sent in doc_question.sents:
                #     roots_question =[st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]

                # p = ''

                # for sent in doc_context.sents:

                #     p += str(sent) + ' '

                #     # print('Sent: ', sent)

                #     doc = nlp(str(sent))

                #     roots_context = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]

                #     # context_name_ents = [str(ent) for ent in doc.ents]

                #     added = False

                #     # print(sent)

                #     for root in roots_question:
                #         if root in roots_context:
                #             temp += str(sent) + ' '
                #             added = True
                #             # print('Here 1')
                #             break

                #     # if not added:
                #     #     for q_ent in question_name_ents:
                #     #         for c_ent in context_name_ents:
                #     #             if q_ent in c_ent or c_ent in q_ent:
                #     #                 temp += str(sent) + ' '
                #     #                 added = True
                #     #                 # print('Here 2')
                #     #                 break
                #     #         if added:
                #     #             break

                #     if not added:
                #         tmp = str(sent)
                #         tmp = tmp.split(' ')
                #         tmp = [PAD_TOKEN] * len(tmp)
                #         tmp = ' '.join(tmp)
                #         temp += tmp + ' '
                #         # print(tmp)
                #         # print('Here 3')

                #     # print('QEnt: ', question_name_ents)
                #     # print('CEnt: ', context_name_ents)
                #     # print('\n\n')

                #     # if not added:
                #     #     if len(context_name_ents) > 0:
                #     #         temp += str(sent) + ' '

                # # print('Debug: ', p[:-1])
                # temp = temp[0: - 1]
                # # print('t1: ', temp)

                # temp = temp.split(' ')
                # # print('t2: ', temp)

                # passage_final = [chunk.lower() for chunk in temp][:self.args.max_context_length]

                # print('Final Passage: ', ' '.join(passage_final))
                # print('QA: ', qa['question'])
                # print('Ans: ', qa['answers'])
                # print('\n\n')
                # print(passage_final)


                # if count == 10:
                #     raise RuntimeError('Debug')

                qid = qa['qid']
                

                question_not_lower = [
                    token for (token, offset) in qa['question_tokens']
                ][:self.args.max_question_length]

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]
                samples.append(
                    (qid, passage, question, answer_start, answer_end, passage_not_lower, question_not_lower)
                )
                # print('Question: ', qa['question'])
                # print('Start: ', answer_start)
                # print('End: ', answer_end)
                # print('Answer: ', qa['answers'])
                # print('\n\n')
            count += 1
                
        return samples

    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        start_positions = []
        end_positions = []
        passages_not_lower = []
        questions_not_lower = []
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid, passage, question, answer_start, answer_end, passage_not_lower, question_not_lower = self.samples[idx]

            # Convert words to tensor.
            passage_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(passage)
            )
            question_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(question)
            )
            answer_start_ids = torch.tensor(answer_start)
            answer_end_ids = torch.tensor(answer_end)

            # Store each part in an independent list.
            passages.append(passage_ids)
            questions.append(question_ids)
            start_positions.append(answer_start_ids)
            end_positions.append(answer_end_ids)
            passages_not_lower.append(passage_not_lower)
            questions_not_lower.append(question_not_lower)

        return zip(passages, questions, start_positions, end_positions, passages_not_lower, questions_not_lower)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            passages_not_lower = []
            questions_not_lower = []
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])
                start_positions[ii] = current_batch[ii][2]
                end_positions[ii] = current_batch[ii][3]
                passages_not_lower.append(current_batch[ii][4])
                questions_not_lower.append(current_batch[ii][5])
                max_passage_length = max(
                    max_passage_length, len(current_batch[ii][0])
                )
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages = torch.zeros(bsz, max_passage_length)
            padded_questions = torch.zeros(bsz, max_question_length)
            # Pad passages and questions
            for iii, passage_question in enumerate(zip(passages, questions)):
                passage, question = passage_question
                padded_passages[iii][:len(passage)] = passage
                padded_questions[iii][:len(question)] = question

            # Create an input dictionary
            batch_dict = {
                'passages': cuda(self.args, padded_passages).long(),
                'questions': cuda(self.args, padded_questions).long(),
                'start_positions': cuda(self.args, start_positions).long(),
                'end_positions': cuda(self.args, end_positions).long(),
                'passages_not_lower': passages_not_lower,
                'questions_not_lower': questions_not_lower
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    def register_tokenizer(self, tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)

# Data loader for BERT

class DataBERT(Dataset):
    def __init__(self, path):
        self.meta, self.elems = load_dataset(path)
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
        self.max_len = 50
        self.data_list = []
        helper(self.data_list, self.elems, self.tokenizer, self.max_len)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def helper(data_list, datas, tokenizer, max_len):
    for data in datas:
        passage = data['context']

        for qa in data['qas']:
            qid = qa['qid']
            question = qa['question']
            start, end = qa['detected_answers'][0]['token_spans'][0]
            answer = qa['detected_answers'][0]['text']

            inputs = tokenizer.encode_plus(question, answer, add_special_tokens = True, max_length = max_len, pad_to_max_length = True)

            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']

            data_list.append((qid, question, start, end, answer, torch.tensor(start), torch.tensor(end), torch.tensor(ids), torch.tensor(mask), torch.tensor(token_type_ids)))

def load_data(path, batch_size = 128):
    dataset = DataBERT(path)
    return DataLoader(dataset, batch_size = batch_size, shuffle = True)