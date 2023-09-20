from analysis.network_helpers import *


def get_adjacency(
        non_overlap: int = 20,
        overlap: int = 0,
        num_blocks: int = 10,
        weighted: bool = False,
        mode: str = 'RAND',
        random_state: Union[int, np.random.RandomState] = 42,
        **kwargs, ):
    _allowed_modes = ['RAND', 'M', 'HM']
    if mode == 'RAND':
        settings = {
            'thresh_low': 0.7,
            'thresh_high': 0.7,
            'thresh_bg': 0.7,
        }
    elif mode == 'M':
        settings = {
            'thresh_low': 0.55,
            'thresh_high': 0.9,
            'thresh_bg': 0.9,
        }
    elif mode == 'HM':
        settings = {
            'thresh_low': 0.45,
            'thresh_high': 0.87,
            'thresh_bg': 0.97,
        }
    else:
        raise ValueError("invalid mode '{}'. available options: {}".format(mode, _allowed_modes))

    for k in settings:
        if k in kwargs:
            settings[k] = kwargs[k]

    rng = get_rng(random_state)

    non_overlap += overlap
    size = non_overlap * 2 * num_blocks
    a = mk_block(size, settings['thresh_bg'], weighted, rng)

    # build
    for h in range(num_blocks):
        main_block = mk_block(non_overlap * 2, settings['thresh_high'], weighted, rng)
        main_block[:non_overlap, :non_overlap] = mk_block(non_overlap, settings['thresh_low'], weighted, rng)
        main_block[non_overlap:, non_overlap:] = mk_block(non_overlap, settings['thresh_low'], weighted, rng)

        start = non_overlap * 2 * h
        end = non_overlap * 2 * (h + 1)
        a[start:end, start:end] = main_block

        if settings['thresh_bg'] > settings['thresh_high']:
            sparse_block = mk_block(non_overlap * 2, settings['thresh_high'], weighted, rng)

            if h < num_blocks - 1:
                a[start: end, end: end + non_overlap * 2] = sparse_block
                a[end: end + non_overlap * 2, start:end] = sparse_block

            else:
                a[start: end, 0: non_overlap * 2] = sparse_block
                a[0: non_overlap * 2, start:end] = sparse_block

    # add overlap to blocks
    if overlap > 0:
        # i == 0 case only
        overlap_block = mk_block(non_overlap * 2 + overlap, settings['thresh_low'], weighted, rng)
        a[-non_overlap:, :overlap] = overlap_block[-non_overlap:, non_overlap: non_overlap + overlap]
        a[-overlap:, :non_overlap] = overlap_block[non_overlap: non_overlap + overlap, -non_overlap:]
        a[:overlap, -non_overlap:] = overlap_block[non_overlap - overlap: non_overlap, :non_overlap]
        a[:non_overlap, -overlap:] = overlap_block[:non_overlap, non_overlap - overlap: non_overlap]

        for i in range(1, num_blocks * 2):
            overlap_block = mk_block(non_overlap * 2 + overlap, settings['thresh_low'], weighted, rng)

            start_short = non_overlap * i - overlap
            end_short = non_overlap * i + overlap
            start_long = non_overlap * (i - 1)
            end_long = non_overlap * (i + 1)
            if i < num_blocks * 2 - 1:
                end_long += overlap

            a[start_short:end_short, start_long:end_long] = \
                overlap_block[non_overlap - overlap: non_overlap + overlap, range(end_long - start_long)]
            a[start_long:end_long, start_short:end_short] = \
                overlap_block[range(end_long - start_long), non_overlap - overlap: non_overlap + overlap]

    communities, memberships, partitions = get_ground_truth(
        width=non_overlap + overlap,
        overlap=overlap,
        num_blocks=num_blocks,
        random_state=random_state,
    )
    return a, communities, memberships, partitions


def mk_block(
        size: int,
        thresh: float,
        weighted: bool = False,
        random_state: int = 42, ):
    if isinstance(random_state, int):
        rng = get_rng(random_state)
    else:
        rng = random_state

    a = rng.uniform(size=(size,) * 2)
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.)

    a[a <= thresh] = 0
    if not weighted:
        a[a > thresh] = 1

    return a.astype(float)


def get_ground_truth(
        width: int,
        overlap: int,
        num_blocks: int,
        random_state: Union[int, np.random.RandomState] = 42, ):

    size = (width - overlap) * 2 * num_blocks

    community = {}
    for i in range(num_blocks * 2):
        start = (width - overlap) * i - overlap
        end = start + width + overlap
        community[i] = [item % size for item in range(start, end)]

    membership = convert('community', 'membership', community, random_state=random_state)
    partition = convert('community', 'partition', community, random_state=random_state)

    return community, membership, partition


def get_cartoon_types(
        size: int,
        ovp_ratio: int = 3,
        thresh_bg: float = 0.85,
        thresh_base: float = 0.60,
        thresh_dense: float = 0.45,
        thresh_sparse: float = 0.70, ):
    size_ovp = size // ovp_ratio

    disj = mk_block(2 * size, thresh_bg)
    dense = mk_block(2 * size, thresh_bg)
    sparse = mk_block(2 * size, thresh_bg)

    for i in range(2):
        s_ = slice(i * size, (i + 1) * size)
        disj[s_][:, s_] = mk_block(size, thresh_base, random_state=i)

        a = i * (size - size_ovp)
        b = (i + 1) * size + (1 - i) * size_ovp
        s_ = slice(a, b)
        dense[s_][:, s_] = mk_block(b - a, thresh_base, random_state=i)
        sparse[s_][:, s_] = mk_block(b - a, thresh_base, random_state=i)

    s_ = slice(size - size_ovp, size + size_ovp)
    dense[s_][:, s_] = mk_block(2 * size_ovp, thresh_dense, random_state=10)
    sparse[s_][:, s_] = mk_block(2 * size_ovp, thresh_sparse, random_state=20)

    output = {
        'disj': disj,
        'dense': dense,
        'sparse': sparse,
    }
    return output
