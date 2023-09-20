from .network import *
from .network_helpers import convert
from .svinet import load_svinet_runs


def compile_lfr_results(lfr_dir: str):
	kw = ['linksampling']
	results = collections.defaultdict(dict)
	results_dir = pjoin(lfr_dir, 'results')
	pbar = tqdm(sorted(os.listdir(results_dir)))
	for item in pbar:
		pbar.set_description(item)
		path = pjoin(results_dir, item)
		lfr = load_lfr(
			load_dir=path,
			weighted=False,
			hierarchical=False,
			verbose=False,
		)
		if not lfr:
			continue
		true_k = len(lfr['community'])

		runs = next(os.walk(path))[1]
		runs = sorted(runs, key=seed_sort_fn)

		_runs = {}
		for run_name in runs:
			if not all(e in run_name for e in kw + [f"k{true_k}"]):
				continue
			num_nodes = re.sub(r"\D", "", run_name.split('-')[0])
			num_nodes = int(num_nodes)
			try:
				_runs[run_name] = load_svinet_runs(
					path=pjoin(path, run_name),
					true_nodes=range(1, num_nodes + 1),
				)
			except FileNotFoundError:
				continue

		gam_all = []
		for i, run in enumerate(_runs.values()):
			try:
				gam_all.append(np.expand_dims(
					run['gam'], axis=0
				))
			except ValueError:
				gam_all.append(np.expand_dims(
					run['gam'][:true_k], axis=0
				))
				gam_all[i] = run['gam'][:true_k]
		gam_all = np.concatenate(gam_all)

		centroids, _ = fit_kmeans(
			data=gam_all.reshape(len(_runs) * true_k, -1),
			n_clusters=true_k,
			match_metric='euclidean',
			random_state=42,
			normalize=True,
		)
		perms = find_best_mappings(
			centroids=Normalizer('l2').fit_transform(centroids),
			data={
				i: Normalizer('l2').fit_transform(gam_all[i])
				for i in range(len(_runs))},
			match_metric='euclidean',
			global_order='infer',
		)
		a, b = [], []
		for i, mapping in perms.items():
			a += [i] * true_k
			b += list(mapping.values())
		gam_all = gam_all[a, b].reshape(gam_all.shape)

		info, *key = item.split('_')
		results[info].update({'_'.join(key): gam_all})
	results = dict(sorted(
		results.items(),
		key=lambda t: int(re.findall(r"\d+", t[0].split('-')[0]).pop())
	))
	return results


def mk_lfr(
		name: str,
		lfr_dir: str,
		degs: Dict[str, Iterable],
		of: float = 0,
		om: int = 0,
		mu: float = 0.1,
		t1: float = 2.0,
		t2: float = 1.0,
		minc: float = 0.05,
		maxc: float = 0.35,
		delim: str = ' ', ):
	if name is None:
		name = 'LFR-disjoint' if of == 0 else 'LFR-overlap'
	network_data = {}
	for key, deg in degs.items():
		nn = np.isfinite(deg).sum()
		lines = [
			delim.join(['-N', f"{nn}\n"]),
			delim.join(['-k', f"{int(np.round(bn.nanmean(deg)))}\n"]),
			delim.join(['-maxk', f"{int(bn.nanmax(deg))}\n"]),
		]
		_lines = [
			delim.join(['-mu', f"{mu}\n"]),
			delim.join(['-t1', f"{t1}\n"]),
			delim.join(['-t2', f"{t2}\n"]),
			delim.join(['-minc', f"{int(np.round(minc * nn))}\n"]),
			delim.join(['-maxc', f"{int(np.round(maxc * nn))}\n"]),
			delim.join(['-on', f"{int(np.round(of * nn))}\n"]),
			delim.join(['-om', f"{om}"]),
		]
		network_data['_'.join([name, key])] = lines + _lines
	_save_lfr_flags(lfr_dir, network_data)
	return


def _save_lfr_flags(lfr_dir: str, network_data: Dict[str, list]):
	for name, v in network_data.items():
		save_dir = pjoin(lfr_dir, 'results', name)
		os.makedirs(save_dir, exist_ok=True)
		with open(pjoin(save_dir, 'flags.dat'), 'w') as f:
			f.writelines(v)
	return


def extract_true_k(lfr_dir: str):
	results_dir = pjoin(lfr_dir, 'results')
	for item in sorted(os.listdir(results_dir)):
		_dir = pjoin(results_dir, item)
		_lfr = load_lfr(
			load_dir=_dir,
			weighted=False,
			hierarchical=False,
			verbose=False,
		)
		if not _lfr:
			print(f'Warning, not found:  {item}')
			continue
		with open(pjoin(_dir, 'true_k.dat'), 'w') as f:
			f.write(str(len(_lfr['community'])))
	return


def load_lfr(
		load_dir: str,
		weighted: bool = False,
		hierarchical: bool = False,
		random_state: int = 42,
		verbose: bool = True, ):

	# load network
	network_file = pjoin(load_dir, 'network.dat')
	try:
		g = nx.read_weighted_edgelist(network_file, nodetype=int)
	except FileNotFoundError:
		return

	# make ordered
	sg = nx.OrderedGraph()
	sg.add_nodes_from(sorted(g.nodes()))
	sg.add_edges_from(
		(u, v, d) for (u, v, d)
		in g.edges(data=True)
		if u in sg if v in sg
	)

	# relable nodes to start indexing from 0
	g = nx.relabel_nodes(
		G=sg,
		mapping={
			i: i - 1 for i in
			range(1, sg.order() + 1)},
		copy=True,
	)

	# normalize weights to [0, 1]
	if weighted:
		max_w = -1
		for u, v, d in g.edges(data=True):
			if d['weight'] >= max_w:
				max_w = d['weight']
		for u, v, d in g.edges(data=True):
			d['weight'] /= max_w

	# load memberships
	if hierarchical:
		file_name = 'community_first_level.dat'
	else:
		file_name = 'community.dat'
	with open(pjoin(load_dir, file_name), 'r') as f:
		lines = f.readlines()

	membership = dict()
	for line in lines:
		node, mem = line.strip('\n').split('\t')
		node = int(node) - 1
		membership[node] = list(
			int(x) - 1 for x in
			filter(None, mem.split(' '))
		)

	# get communities and partition_true
	community = convert(
		conver_from='membership',
		conver_to='community',
		membership=membership,
		random_state=random_state,
	)
	partition = convert(
		conver_from='membership',
		conver_to='partition',
		membership=membership,
		random_state=random_state,
	)

	# get num overlapping nodes
	num_overlap_nodes = len(list(filter(
		lambda x: len(x) > 1, membership.values()
	)))

	if verbose:
		msg = 'Num true communities:\t{:d}\n'
		msg += 'Num overlapping nodes:\t{:d}\n'
		print(msg.format(len(community), num_overlap_nodes))

	lfr = {
		'g': g,
		'a': nx.adjacency_matrix(g).toarray(),
		'community': community,
		'membership': membership,
		'partition': partition,
	}
	return lfr
