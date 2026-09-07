import numpy as np


def get_overlap_between_index_arrays(indices_a, indices_b):
    a = np.ascontiguousarray(indices_a)
    b = np.ascontiguousarray(indices_b)
    a_view = a.view([("", a.dtype)] * a.shape[1]).ravel()
    b_view = b.view([("", b.dtype)] * b.shape[1]).ravel()

    mask = np.isin(a_view, b_view)
    assert np.allclose(mask.shape, (a.shape[0],))  # sanity check

    return mask.sum(), mask.shape[0]


def get_overlap_between_structures(a, b):
    indices_a = np.ascontiguousarray(np.vstack(a.indices(subtree=True)).T)
    indices_b = np.ascontiguousarray(np.vstack(b.indices(subtree=True)).T)
    return get_overlap_between_index_arrays(indices_a, indices_b)


def find_structure_with_most_overlap_in_layer(structure_a, layer_b):
    overlaps = [
        get_overlap_between_structures(structure_a, structure_b)
        for structure_b in layer_b
    ]
    if len(overlaps) == 0:
        return None, (0, len(structure_a.values(subtree=True)))
    most_overlap_with = np.argmax([me[0] for me in overlaps])
    return layer_b[most_overlap_with], overlaps[most_overlap_with]


def _recursive_comparison_function(layer_a, layer_b, res=None):
    if res is None:
        res = {}

    for structure_a in layer_a:
        structure_b, overlap = find_structure_with_most_overlap_in_layer(
            structure_a, layer_b
        )

        level = structure_a.level
        if structure_b is not None:
            assert level == structure_b.level  # sanity check
        if level not in res.keys():
            res[level] = []
        res[level] += [
            overlap,
        ]

        if structure_a.children is not None:
            structure_b_children = (
                structure_b._children if structure_b is not None else []
            )
            _recursive_comparison_function(
                structure_a.children, structure_b_children, res=res
            )

    return res


def compare_dendrograms(referenence, approximation):
    all_overlaps = _recursive_comparison_function(
        referenence.trunk, approximation.trunk
    )
    res = {}
    for level, overlaps in all_overlaps.items():
        _overlapping = np.array([me[0] for me in overlaps])
        _num = np.array([me[1] for me in overlaps])
        mean = _overlapping.sum() / _num.sum()
        res[level] = mean
    return res
