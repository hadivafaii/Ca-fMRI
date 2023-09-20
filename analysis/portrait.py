from analysis.network_helpers import *
# noinspection PyUnresolvedReferences
from portrait_divergence import portrait_cpp, portrait_divergence


def calculate_portraits(
		adj: np.ndarray,
		thres: Iterable[float],
		save_dir: str = None, ):
	portraits = {}
	nonzero = np.abs(adj).mean(0) > 0
	for i, th in enumerate(thres):
		adj_eff = adj[nonzero][:, nonzero]
		adj_eff[adj_eff <= th] = 0.
		g = nx.from_numpy_matrix(adj_eff)
		if save_dir is not None:
			fname = f"{'_'.join(save_dir.split('_')[-3:])}_no{i:03d}_th{th:0.7f}"
		else:
			fname = None
		portraits[(i, th)] = portrait_cpp(g, fname)

	if save_dir is not None:
		save_obj(
			obj=portraits,
			file_name='portraits',
			save_dir=save_dir,
			mode='npy',
			verbose=False,
		)
	return portraits


def calculate_portrait_divergece(portraits_list):
	div = np_nans((len(portraits_list), len(portraits_list)))
	for i, j in itertools.combinations(range(len(portraits_list)), 2):
		div[i, j] = portrait_divergence(portraits_list[i], portraits_list[j])
	return div


def density_space(
		start: float = 0.5,
		stop: float = 0.01,
		num: int = 100,
		log: bool = True, ):
	if log:
		return np.logspace(np.log10(start), np.log10(stop), num)
	else:
		return np.linspace(start, stop, num)


def calculate_portraits_parallel(
		adjacencies: np.ndarray,
		thresholds: np.ndarray,
		dirs: Dict[str, str],
		n_jobs: int = -1, ):
	assert len(adjacencies) == len(thresholds) == len(dirs)
	with joblib.parallel_backend('multiprocessing'):
		joblib.Parallel(n_jobs=n_jobs)(
			joblib.delayed(calculate_portraits)(a, th, d)
			for a, th, d in zip(adjacencies, thresholds, dirs.values())
		)
	return


def calculate_portrait_divergece_parallel(
		portraits: dict,
		densities: np.ndarray,
		n_jobs: int = -1, ):
	with joblib.parallel_backend('multiprocessing'):
		div_list = joblib.Parallel(n_jobs=n_jobs)(
			joblib.delayed(calculate_portrait_divergece)
			([v[di] for v in portraits.values()])
			for di in range(len(densities))
		)
	div = np_nans((len(densities), len(portraits), len(portraits)))
	for di in range(len(densities)):
		div[di] = div_list[di]
	return div


def prepare_divergence_dfs(div, densities, ids):
	dlist = []
	for di, d in enumerate(densities):
		for k, _ids in ids['within'].items():
			data = div[di][list(zip(*_ids))]
			dlist.append({
				'subject': [_sub_from_key(k)] * len(data),
				'condition': ['within'] * len(data),
				'density': [d] * len(data),
				'label': [k] * len(data),
				'val': data,
				'_ids': _ids,
			})
		for k, _ids in ids['across'].items():
			data = div[di][list(zip(*_ids))]
			dlist.append({
				'subject': [_sub_from_key(k)] * len(data),
				'condition': ['across'] * len(data),
				'density': [d] * len(data),
				'label': [k] * len(data),
				'val': data,
				'_ids': _ids,
			})
	df = pd.DataFrame(merge_dicts(dlist))
	df_animal = df.groupby(['subject', 'condition', 'density']).mean()
	df_animal.reset_index(inplace=True)

	df_within = df_animal.loc[df_animal['condition'] == 'within']
	df_table = df_within.pivot_table(values='val', index=['subject'], columns='density')
	group_avg = df_within.groupby(['density']).mean()['val']
	best_i = np.argmin(group_avg)
	best_d = group_avg.index[best_i]
	min_div = group_avg.values[best_i]

	output = {
		'full': df,
		'animal': df_animal,
		'within': df_within,
		'table': df_table,
		'group_avg': group_avg,
		'best_i': best_i,
		'best_d': best_d,
		'best_div': min_div,
	}
	return output


def _sub_from_key(k):
	sub = re.findall(r"\d+", k.split('_')[0])
	sub = next(map(int, sub))
	return sub
