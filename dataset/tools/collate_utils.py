import numpy as np
from utils.common import to_tensor

'''
'stack':
    stack the batch as an array, no matter if they are of the same shape.
'pad-stack':
    pad the batch into the same shape, and then stack them
'concat':
    concatenate batch along the specified dimension
'list':
    doing nothing but link the batch as a list, tensorizing them if required

'''


def get_collateFn(setting):

    def collateFn(datas):
        keys = datas[0].keys()
        assert set(keys).issubset(set(setting.keys())),\
            'mismatching collate setting and data'
        for data in datas:
            assert set(data.keys()) == set(keys),\
                'mismatch among data'
        datas = {k: [data[k] for data in datas] for k in keys}

        output = {}
        for k in keys:
            setting_k = setting[k]
            if setting_k['type'] == 'stack':
                result = np.array(datas[k])

            elif setting_k['type'] == 'pad-stack':
                ndim = datas[k][0].ndim
                dims = [max([data.shape[i] for data in datas[k]])
                        for i in range(ndim)]
                result = np.zeros(shape=(len(datas[k]),) + tuple(dims),
                                    dtype=datas[k][0].dtype)
                if 'pad_value' in setting_k:
                    fill_fn = lambda *args: setting_k['pad_value']
                else:
                    fill_fn = setting_k['pad_fn']

                if ndim == 1:
                    for i, data in enumerate(datas[k]):
                        result[i, :data.shape[0]] = data
                        for j in range(data.shape[0], dims[0]):
                            result[i, j] = fill_fn(j)
                elif ndim == 2:
                    for i, data in enumerate(datas[k]):
                        result[i, :data.shape[0], :data.shape[1]] = data
                        for j in range(data.shape[0], dims[0]):
                            for k in range(data.shape[1], dims[1]):
                                result[i, j, k] = fill_fn(j, k)
                else:
                    raise Exception('n-dimension unsupported: %d' % ndim)

            elif setting_k['type'] == 'concat':
                result = np.concatenate(datas[k], axis=setting_k['axis'])

            elif setting_k['type'] == 'list':
                result = datas[k]
                if setting_k['tensor']:
                    result = [to_tensor(data) for data in result]

            output[k] = to_tensor(result) if setting_k['tensor']\
                else result

        return output

    return collateFn
