from utils.plotting import *
from nancorrmp.nancorrmp import NaNCorrMp


def get_contour(
		obj: np.ndarray,
		s1: float = 0.125,
		s2: float = 0.25, ):
	contour = np.zeros(obj.shape)
	idxs = list(unique_idxs(obj))

	for i in idxs:
		x = (obj == i).astype(float)
		x = sp_img.gaussian_filter(x, sigma=s1)

		x = (x > 0).astype(float)
		x = sp_img.gaussian_filter(x, sigma=s2)

		c = np.logical_or(
			sp_img.sobel(x, axis=0),
			sp_img.sobel(x, axis=1),
		)
		contour[c == 1] = 1

	return contour.astype(bool)


def pi_scores(xa, xb, metric: str):
	assert xa.ndim == xb.ndim == 2, \
		"Pi must have shape (num_k x num_nodes)"

	nans = np.isnan(xa * xb).sum(0)
	nonan = np.where(nans == 0)[0]
	xa = xa[:, nonan]
	xb = xb[:, nonan]

	if metric == 'cosine':
		scores = np.diag(dist2sim(sp_dist.cdist(
			XA=xa,
			XB=xb,
			metric='cosine',
		)))
		scores = {
			oc_idx: s for oc_idx, s in
			enumerate(scores, start=1)
		}
	elif metric == 'js':
		scores = np_nans(len(nans))
		_scores = np.diag(dist2sim(sp_dist.cdist(
			XA=xa.T,
			XB=xb.T,
			metric='jensenshannon',
		)))
		scores[nonan] = _scores
		scores = {
			node: s for node, s in
			enumerate(scores, start=0)
		}
	elif metric == 'hamming':
		s = sp_dist.hamming(
			u=np.argmax(xa, axis=0),
			v=np.argmax(xb, axis=0),
		)
		scores = {-1: s}
	else:
		raise NotImplementedError

	if -1 not in scores:
		scores[-1] = bn.nanmean(list(scores.values()))
	return scores


def pi_from_gam(
		gam: np.ndarray,
		axis_comm: int = -2,
		axis_seed: int = -3,
		est_seed: Callable = bn.nanmean, ):
	if est_seed is not None and gam.ndim > 5:
		gam = est_seed(gam, axis=axis_seed)
	pi = gam / np.nansum(gam, axis=axis_comm, keepdims=True)
	return pi


def entropy_expect_beta(arr: np.ndarray, normalize: bool = True):
	from scipy import special
	k, num_nodes = arr.shape[-2:]
	eh = np.zeros((*arr.shape[:-2], num_nodes))

	for node in range(num_nodes):
		g = arr[..., node]
		a0 = g.sum(-1)
		psi0 = special.digamma(a0 + 1)
		psi = special.digamma(g + 1)
		x = (g * psi).sum(-1)
		eh[..., node] = psi0 - x / a0
	if normalize:
		eh /= np.log(k)
	return eh


def cut_data(
		data: np.ndarray,
		bins: np.ndarray,
		decimals: int = 3,
		axis_oc: int = 1,
		axis_node: int = 2,
		axis_batch: int = 0,
		percent: bool = True,
		oc_only: bool = False, ):
	assert data.ndim in [2, 3], "dims: [batch], oc, node"
	x = np.round(data, decimals=decimals)
	if data.ndim == 2:
		cut_oc = _cut(x, 0, bins, percent)
		if not oc_only:
			cut_nodes = _cut(x, 1, bins, percent)
		else:
			cut_nodes = None
	else:
		if axis_batch == -1:
			axis_batch = np.indices((3,))[0][axis_batch]
		a = x.shape[axis_batch]
		b = x.shape[axis_oc]
		c = x.shape[axis_node]

		_shape = (b, len(bins) - 1)
		_shape = np.insert(_shape, axis_batch, a)
		cut_oc = np_nans(tuple(_shape))

		_shape = (c, len(bins) - 1)
		_shape = np.insert(_shape, axis_batch, a)
		cut_nodes = np_nans(tuple(_shape)) if not oc_only else None

		for i in range(a):
			s_ = tuple(
				i if j == axis_batch else np.s_[:]
				for j in range(x.ndim)
			)
			cut_oc[s_] = _cut(
				x=x[s_],
				bins=bins,
				percent=percent,
				axis=int(axis_oc > axis_node),
			)
			if not oc_only:
				cut_nodes[s_] = _cut(
					x=x[s_],
					bins=bins,
					percent=percent,
					axis=int(axis_oc < axis_node),
				)
	return cut_oc, cut_nodes


def _cut(x, axis, bins, percent):
	cut = np_nans((x.shape[axis], len(bins) - 1))
	for i in range(x.shape[axis]):
		s_ = tuple(
			i if j == axis
			else np.s_[:]
			for j in range(x.ndim)
		)
		counts = collections.Counter(np.searchsorted(
			a=bins, v=x[s_], side='left',
		))
		counts = [counts.get(i, 0) for i in range(1, len(bins))]
		tot = sum(counts)
		if tot:
			portions = [c / tot for c in counts]
		else:
			continue
		if percent:
			cut[i] = [e * 100 for e in portions]
		else:
			cut[i] = portions
	return cut


def fit_kmeans(
		data: np.ndarray,
		n_clusters: int,
		match_metric: str,
		random_state: int = 0,
		kw_kmeans: dict = None,
		normalize: bool = True,
		verbose: bool = False, ) -> (np.ndarray, Any):
	if match_metric in ['euclidean', 'sqeuclidean']:
		from sklearn.cluster import KMeans
		import sklearnex
		sklearnex.patch_sklearn(verbose=verbose)
		if normalize:
			data = Normalizer().fit_transform(data)
		km = KMeans(
			n_clusters=n_clusters,
			random_state=random_state,
			**filter_kwargs(KMeans, kw_kmeans),
		).fit(data)
		centroids = km.cluster_centers_.copy()
		sklearnex.unpatch_sklearn()
	else:
		from nltk.cluster.kmeans import KMeansClusterer
		from nltk.cluster.util import cosine_distance
		km = KMeansClusterer(
			num_means=n_clusters,
			distance=cosine_distance if match_metric == 'cosine'
			else get_distance_fn(match_metric),
			rng=get_rng(random_state, use_np=False),
			**filter_kwargs(KMeansClusterer, kw_kmeans),
		)
		km.cluster(data)
		centroids = [e.reshape(1, -1) for e in km.means()]
		centroids = np.concatenate(centroids)
	if verbose:
		print('[PROGRESS] kmeans fit: done')
	return centroids, km


def clusters_sim_scores(
		masks_fx: dict,
		masks_mv: dict,
		weighted: bool = True,
		global_order: bool = True,
		assert_equal: bool = True, ):
	weights = {k: m.sum() for k, m in masks_fx.items()}

	dice = np.zeros((len(masks_fx), len(masks_mv)))
	overlap = np.zeros((len(masks_fx), len(masks_mv)))
	jaccard = np.zeros((len(masks_fx), len(masks_mv)))
	for i, (k, m1) in enumerate(masks_fx.items()):
		for j, m2 in enumerate(masks_mv.values()):
			w = weights[k] / sum(weights.values()) if weighted else 1.0
			dice[i, j] = dice_score(m1, m2) * w
			overlap[i, j] = overlap_score(m1, m2) * w
			jaccard[i, j] = jaccard_score(m1, m2) * w
	scores = {
		'dice': dice,
		'overlap': overlap,
		'jaccard': jaccard,
	}
	mappings = {
		k: find_best_match(
			scores=v,
			reverse=True,
			global_order=global_order,
			assert_equal=assert_equal,
		)
		for k, v in scores.items()
	}
	scores = {
		k: v[:, list(mappings[k].values())]
		for k, v in scores.items()
	}
	return scores, mappings


def find_best_mappings(
		centroids: np.ndarray,
		data: Dict[Any, np.ndarray],
		match_metric: str = 'cosine',
		global_order: Union[str, bool] = 'infer',
		filter_nans: bool = True, ) -> Dict[Any, dict]:
	perms = {}
	for key, x in data.items():
		if filter_nans:
			nans = np.logical_or(
				np.isnan(x).sum(0).astype(bool),
				np.isnan(centroids).sum(0).astype(bool),
			)
			good_ids = np.where(~nans)[0]
		else:
			good_ids = range(centroids.shape[1])
		# noinspection PyTypeChecker
		d = sp_dist.cdist(
			XA=centroids[:, good_ids],
			XB=x[:, good_ids],
			metric=match_metric,
		)
		if match_metric in ['cosine', 'correlation']:
			if global_order == 'infer':
				mapping_1 = find_best_match(
					scores=dist2sim(d),
					reverse=True,
					global_order=True,
				)
				mapping_2 = find_best_match(
					scores=dist2sim(d),
					reverse=True,
					global_order=False,
				)
				d1 = d[:, list(mapping_1.values())]
				d2 = d[:, list(mapping_2.values())]
				d1 = d1.diagonal().mean()
				d2 = d2.diagonal().mean()
				if d1 < d2:
					mapping = mapping_1
				else:
					mapping = mapping_2
			else:
				mapping = find_best_match(
					scores=dist2sim(d),
					reverse=True,
					global_order=global_order,
				)
		else:
			if global_order == 'infer':
				mapping_1 = find_best_match(
					scores=d,
					reverse=False,
					global_order=True,
				)
				mapping_2 = find_best_match(
					scores=d,
					reverse=False,
					global_order=False,
				)
				d1 = d[:, list(mapping_1.values())]
				d2 = d[:, list(mapping_2.values())]
				d1 = d1.diagonal().mean()
				d2 = d2.diagonal().mean()
				if d1 < d2:
					mapping = mapping_1
				else:
					mapping = mapping_2
			else:
				mapping = find_best_match(
					scores=d,
					reverse=False,
					global_order=global_order,
				)
		perms[key] = mapping
	return perms


def find_best_match(
		scores: np.ndarray,
		reverse: bool = True,
		fill_nans: bool = True,
		global_order: bool = True,
		assert_equal: bool = True, ) -> Dict[int, int]:
	n, m = scores.shape
	if fill_nans:
		nans = np.isnan(scores)
		scores[nans] = np.inf * (-1 if reverse else 1)
	if global_order:
		ind = np.argsort(scores.ravel())
		ind_x, ind_y = np.unravel_index(
			indices=ind[::-1] if reverse else ind,
			shape=scores.shape,
		)
		ctr = 0
		ii, jj = [], []
		while True:
			if ind_x[ctr] not in ii and ind_y[ctr] not in jj:
				ii.append(ind_x[ctr])
				jj.append(ind_y[ctr])
			ctr += 1
			cond = (
					list(set(ii)) == list(range(n)) and
					list(set(jj)) == list(range(m))
			)
			cond_safety = ctr >= len(scores.ravel())
			if cond or cond_safety:
				break
		mapping = {i: j for i, j in zip(ii, jj)}
		mapping = dict(sorted(mapping.items()))
	else:
		mapping = {}
		available = list(range(m))
		for i in range(n):
			if len(available):
				if reverse:
					j = np.argmax(scores[i][available])
				else:
					j = np.argmin(scores[i][available])
				mapping[i] = available.pop(int(j))
	if assert_equal:
		assert len(mapping) == n, "must find a match for each row"
	return mapping


def vec2vol(vec: Dict[int, Dict[str, float]], mice):
	vols = {}
	for idx, d in vec.items():
		x = np.zeros(mice.al.dims)
		for lbl, v in d.items():
			if len(lbl.split('-')) == 3:
				ii = mice.parcel.encode(lbl)
				x[mice.parcel.results['cortex-merged'] == ii] = v
			else:
				lr, ii = tuple(map(lambda s: int(s), lbl.split('-')))
				if lr == 1:
					x[mice.parcel.results['subcortex-l'] == ii] = v
				else:
					x[mice.parcel.results['subcortex-r'] == ii] = v
		vols[idx] = x.astype(float)
	return vols


def calculate_threshold(
		adj: np.ndarray,
		rho: Union[float, Iterable[float]],
		verbose: bool = False, ):
	if not isinstance(rho, Iterable):
		rho = [rho]
	nan = np.isnan(adj).sum(0) == len(adj)
	zero = np.nanmean(np.abs(adj), 0) == 0
	good = np.logical_and(~nan, ~zero)
	adj_eff = adj[good][:, good]
	n = good.sum()
	m_max = n * (n - 1) / 2
	y = sp_dist.squareform(adj_eff)
	thres = [int(np.round(m_max * d)) for d in rho]
	thres = [sorted(y, reverse=True)[i] for i in thres]
	if verbose:
		for d, th in zip(rho, thres):
			adj_eff = adj[good][:, good]
			adj_eff[adj_eff <= th] = 0.
			g = nx.from_numpy_matrix(adj_eff)
			msg = f"threshold = {th:0.3f}   /   "
			msg += f"density:  requested = {d:0.5f}, "
			msg += f"result = {nx.density(g):0.5f};"
			print(msg)
		print('\n')
	return thres


def calculate_threshold_old(
		adj: np.ndarray,
		rho: Union[float, Iterable[float]],
		verbose: bool = False, ):
	nan = np.isnan(adj).sum(0) == len(adj)
	zero = np.nanmean(np.abs(adj), 0) == 0
	good = np.logical_and(~nan, ~zero)
	adj_eff = adj[good][:, good]
	adj_eff = np.maximum(0, adj_eff)
	y = sp_dist.squareform(adj_eff)
	if not isinstance(rho, Iterable):
		rho = [rho]
	thresholds = np.quantile(y, [1 - d for d in rho])
	if verbose:
		for d, thres in zip(rho, thresholds):
			adj_eff = adj[good][:, good]
			adj_eff[adj_eff <= thres] = 0.
			g = nx.from_numpy_matrix(adj_eff)
			msg = f"threshold = {thres:0.3f}   /   "
			msg += f"density:  requested = {d:0.5f}, "
			msg += f"result = {nx.density(g):0.5f};"
			print(msg)
		print('\n')
	return thresholds


def get_adj_mat(
		x: Union[dict, np.ndarray, pd.DataFrame],
		metric: str, p_val: float = 0.05, ):
	x = _check_input(x)
	if metric == 'spearman':
		# noinspection PyTypeChecker
		a, p = sp_stats.spearmanr(x, axis=1)
	elif metric == 'pearson':
		a, p = NaNCorrMp.calculate_with_p_value(x)
	else:
		raise RuntimeError(f"invalid metric: {metric}")

	a[np.isnan(a)] = 0.
	p[np.isnan(p)] = 1.

	a[p >= p_val] = 0.
	np.fill_diagonal(a, 0.)
	a = (a + a.T) / 2
	return a, p


def get_pdist(
		x: Union[dict, np.ndarray, pd.DataFrame],
		metric: str = 'correlation', ):
	x = _check_input(x)

	cond = metric in ['correlation', 'pearson', 'spearman']
	if np.sum(~np.isfinite(x)) and cond:
		if metric == 'spearman':
			a, p = sp_stats.spearmanr(x, axis=1)
		else:
			a, p = NaNCorrMp.calculate_with_p_value(x)
		a[np.isnan(a)] = 0.
		p[np.isnan(p)] = 1.
		a[p >= 0.05] = 0.
		d = 1 - (a + a.T) / 2
		np.fill_diagonal(d, 0.)
		assert not (d != d.T).sum()
		y = sp_dist.squareform(d)
	else:
		# noinspection PyTypeChecker
		y = sp_dist.pdist(x, metric=metric)
	output = {
		'x': x, 'y': y,
		'y_sq': sp_dist.squareform(y),
	}
	return output


def get_distance_fn(metric):
	if metric == 'cosine':
		return sp_dist.cosine
	elif metric == 'correlation':
		return sp_dist.correlation
	elif metric == 'jensenshannon':
		return sp_dist.jensenshannon
	elif metric == 'euclidean':
		return sp_dist.euclidean
	elif metric == 'sqeuclidean':
		return sp_dist.sqeuclidean
	elif metric == 'canberra':
		return sp_dist.canberra
	elif metric == 'braycurtis':
		return sp_dist.braycurtis
	elif metric == 'chebyshev':
		return sp_dist.chebyshev
	elif metric == 'minkowski':
		return sp_dist.minkowski
	else:
		err = f"metric '{metric}' not supported"
		raise ValueError(err)


# TODO: add that other function dist2adj
def dist2sim(d: np.ndarray) -> np.ndarray:
	if not np.all(d >= 0):
		msg = f"Warning, '{np.isnan(d).sum()}' "
		msg += 'nans detected . . . filling with ones'
		print(msg)
		d[np.isnan(d)] = 1
	assert np.all(d >= 0)
	if 0 <= np.max(d) <= 2:
		s = 1 - d
	else:
		s = 1 / np.maximum(d, np.finfo(float).eps)
		s /= np.max(s)
	return s


def weighted_score(values, masks):
	numer = [m.sum() * score for m, score in zip(masks.values(), values)]
	denum = [m.sum() for m in masks.values()]
	return sum(numer) / sum(denum)


def symmetry_score(m, mask_l, mask_r):
	ll = np.logical_and(m.astype(bool), mask_l.astype(bool)).sum()
	rr = np.logical_and(m.astype(bool), mask_r.astype(bool)).sum()
	if max(ll, rr) == 0:
		return np.nan
	else:
		return 1 - abs(ll - rr) / max(ll, rr)


def dice_score(m1, m2):
	numer = np.logical_and(m1.astype(bool), m2.astype(bool)).sum()
	denum = np.sum(m1.astype(bool)) + np.sum(m2.astype(bool))
	return 2.0 * numer / denum


def overlap_score(m1, m2):
	numer = np.logical_and(m1.astype(bool), m2.astype(bool)).sum()
	denum = min(m1.astype(bool).sum(), m2.astype(bool).sum())
	return numer / denum


def overlap_score_idxs(i1, i2):
	numer = len(set(i1).intersection(i2))
	denum = min(len(i1), len(i2))
	return numer / denum


def jaccard_score(m1, m2):
	d = sp_dist.jaccard(m1.ravel().astype(bool), m2.ravel().astype(bool))
	return 1 - d


def seed_sort_fn(x):
	try:
		return next(
			int(e.replace('seed', '').strip())
			for e in x.split('-') if 'seed' in e
		)
	except StopIteration:
		return -1


def sort_date_latest(x):
	x = x.split('(')[-1].split(')')[0].split('_')
	try:
		yy, mm, dd = tuple(map(lambda s: int(s), x))
		return (yy - 2021) * 30 * 365 + mm * 30 + dd
	except ValueError:
		return -1


def _check_input(x):
	if isinstance(x, dict):
		x = [
			np.array(vec).reshape(1, -1)
			for vec in x.values()
		]
		x = np.concatenate(x)
	elif isinstance(x, pd.DataFrame):
		x = x.values.T
	elif isinstance(x, np.ndarray):
		pass
	else:
		msg = 'invalid input type: '
		msg += f"'{type(x).__name__}'"
		raise ValueError(msg)
	return x
