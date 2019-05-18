import json
import os

from . import question_utils


class Protocol:
    def __init__(self, allow_output_protocol, protocol_file, gather=False, use_special_tokens=False):
        self.allow_output_protocol = allow_output_protocol
        self.gather = gather
        self.protocol_file = protocol_file
        self.use_special_tokens = use_special_tokens

        if os.path.exists(protocol_file):
            with open(self.protocol_file, 'r') as f:
                self.records_ = json.load(f)
        else:
            self.records_ = {}

        if gather:
            self.records_['total'] = []

        if not gather:
            self.records2idx = {key: {k: i for i, k in enumerate(self.records_[key])}
                                for key in self.records_}
        else:
            self.records2idx = {k: i for i, k in enumerate(self.records_['total'])}

    def reset(self):
        self.records_ = {}
        if os.path.exists(self.protocol_file):
            with open(self.protocol_file, 'r') as f:
                self.records_ = json.load(f)
        else:
            if self.gather:
                self.records_['total'] = []

        if not self.gather:
            self.records2idx = {key: {k: i for i, k in enumerate(self.records_[key])}
                                for key in self.records_}
        else:
            self.records2idx = {k: i for i, k in enumerate(self.records_['total'])}

    def __getitem__(self, query):
        if isinstance(query, tuple):
            category, item = query
            if not category in self.records_:
                self._add_record(category)
            if isinstance(item, int):
                if item < len(self.records_[category]):
                    return self.records_[category][item]
                else:
                    return '<UNK>'
            else:
                if item in self.records_[category]:
                    if not self.gather:
                        return self.records2idx[category][item]
                    else:
                        return self.records2idx[item]
                else:
                    self._add_element(category, item)
                    return self[category, item]
        else:
            if self.gather and query in self.records2idx:
                return self.records2idx[query]
            else:
                category = query
                if not category in self.records_:
                    self._add_record(category)
                return self.records_[category]

    def _add_element(self, category, item):
        self.records_[category].append(item)
        if self.gather:
            if not item in self.records_['total']:
                self.records_['total'].append(item)

        if not self.gather:
            self.records2idx[category].update(
                {item: self.records_[category].index(item)})
        else:
            self.records2idx.update({item: self.records_['total'].index(item)})

        if self.allow_output_protocol and\
                os.path.isdir(os.path.dirname(self.protocol_file)):
            with open(self.protocol_file, 'w') as f:
                json.dump(self.records_, f)

    def _add_record(self, category):
        self.records_[category] = []
        if not self.gather:
            self.records2idx[category] = {}
        if self.use_special_tokens:
            for v in list(question_utils.special_tokens):
                self._add_element(category, v)

    @property
    def records(self):
        return {k:v for k, v in self.records_.items()
                if k!='total'}

    def belongs_to(self, name):
        cats = [cat for cat, content in self.records.items()
                if name in content and cat != 'classes']
        if len(cats) > 1:
            raise Exception('%s belongs to multiple categories')
        else:
            return cats[0]

    @property
    def concepts(self):
        names = []
        for cat, attributes in sorted(self.records.items()):
            names += sorted(attributes)
        return names
