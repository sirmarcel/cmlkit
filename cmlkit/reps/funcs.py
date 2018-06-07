import numpy as np
import qmmlpack as qmml
from cmlkit.helpers import convert_sequence


def explicit_single_mbtr(z, r, b, d, k, geomf, weightf, distrf, corrf, eindexf, aindexf, flatten, elems, acc):
    kgwdcea = [k, geomf, weightf, distrf, corrf, eindexf, aindexf]

    return qmml.many_body_tensor(z, r, d, kgwdcea, flatten=flatten, elems=elems, basis=b, acc=acc)


def explicit_single_mbtr_with_norm(norm, z, r, b, d,
                                   k, geomf, weightf, distrf, corrf, eindexf, aindexf,
                                   flatten, elems, acc,
                                   mbtr_gen=explicit_single_mbtr):

    m = mbtr_gen(z, r, b, d, k, geomf, weightf, distrf, corrf, eindexf, aindexf, flatten, elems, acc)

    if norm is not None:
        for i in range(len(z)):
            m[i] /= np.sum(m[i])
        return m * norm
    else:
        return m


def single_mbtr(dataset, single_mbtr_spec, mbtr_gen=explicit_single_mbtr_with_norm):
    geomf = convert_sequence(single_mbtr_spec['geomf'])
    weightf = convert_sequence(single_mbtr_spec['weightf'])
    distrf = convert_sequence(single_mbtr_spec['distrf'])
    corrf = convert_sequence(single_mbtr_spec['corrf'])
    eindexf = convert_sequence(single_mbtr_spec['eindexf'])
    aindexf = convert_sequence(single_mbtr_spec['aindexf'])

    d = tuple(single_mbtr_spec['d'])

    return mbtr_gen(single_mbtr_spec['norm'],
                    dataset.z, dataset.r, dataset.b,
                    d, single_mbtr_spec['k'],
                    geomf, weightf, distrf, corrf, eindexf, aindexf,
                    single_mbtr_spec['flatten'], single_mbtr_spec['elems'], single_mbtr_spec['acc'])


def make_mbtrs(dataset, spec_mbtrs, mbtr_gen=single_mbtr):
    return np.concatenate([mbtr_gen(dataset, args) for k, args in spec_mbtrs.items()], axis=1)
