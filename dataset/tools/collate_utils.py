import numpy as np
from utils.common import to_tensor

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
                result_np = np.array(datas[k])

            elif setting_k['type'] == 'pad-stack':
                ndim = datas[k][0].ndim
                dims = [max([data.shape[i] for data in datas[k]])
                        for i in range(ndim)]
                result_np = np.zeros(shape=(len(datas[k]),) + tuple(dims),
                                    dtype=datas[k][0].dtype)
                if 'pad_value' in setting_k:
                    fill_fn = lambda *args: setting_k['pad_value']
                else:
                    fill_fn = setting_k['pad_fn']

                if ndim == 1:
                    for i, data in enumerate(datas[k]):
                        result_np[i, :data.shape[0]] = data
                        for j in range(data.shape[0], dims[0]):
                            result_np[i, j] = fill_fn(j)
                elif ndim == 2:
                    for i, data in enumerate(datas[k]):
                        result_np[i, :data.shape[0], :data.shape[1]] = data
                        for j in range(data.shape[0], dims[0]):
                            for k in range(data.shape[1], dims[1]):
                                result_np[i, j, k] = fill_fn(j, k)
                else:
                    raise Exception('n-dimension unsupported: %d' % ndim)

            elif setting_k['type'] == 'concat':
                result_np = np.concatenate(datas[k], axis=setting_k['axis'])

            output[k] = to_tensor(result_np) if setting_k['tensor']\
                else result_np

        return output

    return collateFn
