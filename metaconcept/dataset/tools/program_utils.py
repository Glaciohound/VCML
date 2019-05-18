def preprocess_operation(cur):
    operation = cur['operation']
    category = None
    if ' ' in operation:
        splits = operation.split(' ')
        if len(splits) > 2:
            operation = splits[0]
            # category = ''.join([splits[1]] + [s[0].upper()+s[1:] for s in splits[2:]])
            category = '_'.join(splits[1:])
        else:
            operation, category = splits
    if category == 'rel':
        category = 'relation'

    argument = cur['argument']
    if ' (' in argument:
        argument, obj = argument.split(' (')
        obj = obj.split(')')[0].split(',')
    elif argument == 'scene':
        obj = 'scene'
    else:
        obj = None

    argument = argument[:-1] if argument.endswith(' ') else argument
    if argument == '' and not operation in ['and', 'or', 'same', 'different']:
        argument = 'scene'
    elif argument == '?':
        argument = ''
    #elif ' ' in argument:
    #    argument = argument.replace(' ', '_')
    # TODO: need checking

    return operation, category, argument, obj


def semantic2program_r(program_list):
    result = []

    def set_scene():
        set_mode('new_scene')
        result.append({
            'operation': 'insert',
            'argument': 'scene',
        })

    def set_insert(argument, transfer=True):
        if isinstance(argument, list):
            for item in argument:
                set_insert(item, transfer=False)
            set_transfer()
            return
        else:
            append_ = {
                'operation': 'insert'
            }
            if isinstance(argument, str):
                kwargs = {'argument': argument}
            else:
                kwargs = argument
            append_.update(kwargs)
            result.append(append_)
        if transfer:
            set_transfer()

    def set_transfer():
        result.append({
            'operation': 'transfer',
            'argument': '',
        })

    def set_mode(mode):
        result.append({
            'operation': 'mode',
            'argument': mode,
        })

    def set_same(argument, category):
        argument, inputType = (argument, 'same_among') if argument != '' else (category, 'same_between')
        if inputType == 'same_between':
            set_mode('both')
        set_insert([argument, 'attr_only'])
        set_insert(['is_unique'])

    def set_not():
        set_insert('boolean_not')

    def set_filter(category, argument, not_=False):
        # TODO how to utilize the category information here?
        append_ = [{'argument': argument, 'category': category}]
        if not_:
            append_.append('not')
        set_insert(append_)
        set_insert('object_only')

    def set_logical(operation):
        set_mode('both')
        set_insert(operation)

    def set_select(argument, obj=None):
        # TODO how to utilize the category information here?
        set_scene()
        if argument not in ['_', 'this']:
            append_ = {'argument': argument,
                       'category': 'name',
                       'object': obj}
            if obj == None:
                append_.pop('object')
            set_insert(append_)

    def set_choose(category, argument):
        if category == 'relation':
            set_mode('both')
        arg1, arg2 = argument.split('|')
        set_insert([{'category': category, 'argument': arg1},
                    {'category': category, 'argument': arg2}])
        set_insert('rel_only' if category == 'relation' else 'attr_only')

    def set_verifyRelation(argument, obj):
        another, argument, category = argument.split(',')
        argument
        set_select(another)

        if category == 'relation':
            set_mode('both')

        set_insert(category)
        set_insert(argument)
        set_insert('rel_exist' if category == 'relation' else 'attr_exist')

    def set_relate(argument, obj):
        another, argument, category = argument.split(',')
        set_insert([category, argument])
        set_insert(another)

    def build_list(cur):
        operation, category, argument, obj = preprocess_operation(cur)

        for i in cur['dependencies']:
            build_list(program_list[i])

        if argument.startswith('not('):
            # not filter operations
            argument = argument[4:-1]
            set_filter(category, argument, not_=True)

        elif '|' in argument:
            # choose questions
            if category == 'relation':
                # choose relation
                another, argument, category = argument.split(',')
                set_select(another)
            # else choose attribute or name
            set_choose(category, argument)

        elif operation == 'relate':
            # filter by relation
            set_relate(argument, obj)

        elif operation == 'choose':
            # "choose Xer" operations
            set_mode('both')
            set_insert(category)

        elif category == 'relation':
            # verify relation operations
            set_verifyRelation(argument, obj)

        elif operation == 'filter':
            # filter by attributes
            set_filter(category, argument)

        elif operation == 'select':
            # select by names
            set_select(argument, obj)

        elif operation == 'same':
            # same {type, gender, color, ...} operations
            set_same(argument, category)

        elif operation == 'different':
            # different {type, gender, color, ...} operations
            set_same(argument, category)
            set_not()

        elif operation in ['and', 'or']:
            # logical operations
            set_logical(operation)

        elif operation == 'query':
            # query attribute
            set_insert(argument)

        elif operation == 'exist':
            # whether a kind of objects exist
            set_insert(operation)

    build_list(program_list[-1])
    return result


def tree2postfix(program_tree):
    output = []

    def helper(current):
        for node in current['dependencies']:
            helper(node)
        output.append(current)

    helper(program_tree)
    for node in output:
        if 'dependencies' in node:
            node.pop('dependencies')
    return output


def function2str(f):
    return 'Function[%s, %s](%s)' % (
        f['operation'],
        f['category'],
        f['argument']
    )


def function2compactStr(f):
    category = f['category'] if f['category'] in ['s', 'o', 'same1', 'same2'] else ''

    return '(%s%s%s%s%s)' % (
        f['operation'],
        '_' if category else '',
        category,
        ', ' if f['argument'] else '',
        f['argument'],
    )


''' semantic translator for u_embedding'''


def semantic2program_u(program_list):
    output = []

    def add_operation(x, y):
        output.append({'operation': x,
                       'argument': y})

    def convert_operation(x):
        output[-1]['operation'] = x

    for op in program_list:
        operation, category, argument, obj = preprocess_operation(op)

        if operation == 'select':
            add_operation('transfer', 'object_only')
            add_operation('verify', argument)

        elif operation == 'filter':
            add_operation('verify', argument)

        elif operation == 'query':
            add_operation('transfer', argument)

        elif operation == 'exist':
            convert_operation('transfer')

        elif operation == 'select_concept':
            add_operation('transfer', 'concept_only')
            add_operation('verify', argument)

        elif operation == 'synonym':
            add_operation('transfer', 'synonym')
            add_operation('transfer', argument)

        elif operation == 'classify':
            add_operation('classify', argument)

        else:
            raise Exception('no such operation supported {}'.format(op))

    return output


''' semantic translator for h_embedding'''


def semantic2program_h(program_list):
    output = []

    def add_operation(x, y):
        output.append({'operation': x,
                       'argument': y})

    def convert_operation(x):
        output[-1]['operation'] = x

    for op in program_list:
        operation, category, argument, obj = preprocess_operation(op)

        if operation == 'select':
            add_operation('select', 'object_only')
            add_operation('filter', argument)

        elif operation == 'select_concept':
            add_operation('select', 'concept_only')
            add_operation('choose', argument)

        elif operation == 'filter':
            add_operation('filter', argument)

        elif operation == 'query':
            add_operation('transfer_oc', argument)

        elif operation == 'isinstance':
            add_operation('transfer_cc', operation)

        elif operation == 'synonym':
            add_operation('transfer_cc', operation)
            add_operation('verify', argument)
            add_operation('exist', '<NULL>')

        elif operation == 'exist':
            add_operation('exist', '<NULL>')

        elif operation == 'classify':
            add_operation('classify', argument)

        elif operation == '<NULL>':
            add_operation('<NULL>', '<NULL>')

        else:
            raise Exception('no such operation supported {}'.format(op))

    return output
