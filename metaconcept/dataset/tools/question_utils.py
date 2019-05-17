special_tokens = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}
import numpy as np
from . import program_utils
from metaconcept import info, args


def build_tokenMap(obj, vocabulary, add_special_tokens=False):
    tokenMap = {}
    if add_special_tokens:
        for l in vocabulary.values():
            l += special_tokens
    for c in vocabulary.keys():
        item = vocabulary[c]
        tokenMap[c + '2idx'], tokenMap['idx2' + c] = \
            ({y: i for i, y in enumerate(item)}, item)
    for k, v in tokenMap.items():
        setattr(obj, k, v)


def tokenize(s, delim=' ',
             add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def encode_question(question, allow_unk=False):
    seq_tokens = tokenize(question, punct_to_keep=[';', ',', '?', '.'])
    seq_idx = []
    for token in seq_tokens:
        seq_idx.append(token)
    seq_idx = [info.protocol['words', x] for x in seq_idx]
    return np.array(seq_idx)


def filter_questions(question, mode):
    if mode == 'None':
        return True
    elif mode == 'existance':
        for op in question['semantic']:
            if op['operation'] not in ['exist', 'select']:
                return False
    return True

def register_concepts(questions):
    if isinstance(questions, dict):
        questions = questions.values()

    for q in questions:
        for op in program_utils.semantic2program_h(q['semantic']):
            info.protocol['operations', op['operation']]
            info.protocol['concepts', op['argument']]
            if op['operation'].startswith('transfer'):
                info.protocol['metaconcepts', op['argument']]

        encode_question(q['question'])
        info.protocol['concepts', q['answer']]

    args.max_concepts = max(args.max_concepts,
                            len(info.protocol['concepts']))

    return info.protocol['metaconcepts']
