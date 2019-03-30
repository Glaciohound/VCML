import json
import os
from dataset.tools import question_utils

class Protocol:
    def __init__(self, args, info=None):
        self.args = args
        self.info = info
        self.load_protocol()

    def load_protocol(self):
        args = self.args
        info = self.info

        if os.path.exists(args.protocol_file):
            with open(self.args.protocol_file, 'r') as f:
                self.protocol = json.load(f)
        else:
            self.protocol = {'words': [], 'concepts': [], 'operations': []}
        if info is not None:
            info.protocol = self

        for value in self.protocol.values():
            for token in question_utils.special_tokens:
                if token not in value:
                    value.append(token)

        self.protocol2idx = {key: {k: i for i, k in enumerate(self.protocol[key])}
                             for key in self.protocol}

    def __getitem__(self, query):
        if isinstance(query, tuple):
            category, item = query
            if isinstance(item, int):
                if item < len(self.protocol[category]):
                    return self.protocol[category][item]
                else:
                    raise Exception('unknown token')
            else:
                if item in self.protocol[category]:
                    return self.protocol2idx[category][item]
                else:
                    return self.add_element(category, item)
        else:
            category = query
            if category in self.protocol:
                return self.protocol[category]
            else:
                return self.protocol2idx[category.split('2')[0]]

    def add_element(self, category, item):
        args = self.args
        n = len(self.protocol[category])
        self.protocol[category].append(item)
        self.protocol2idx[category].update({item: n})
        if args.allow_output_protocol and\
                os.path.exists(os.path.join(
                    *args.protocol_file.split('/')[:-1])):
            with open(args.protocol_file, 'w') as f:
                json.dump(self.protocol, f)
        return n
