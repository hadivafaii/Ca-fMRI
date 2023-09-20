from analysis.base import *
from model.configuration import Config
from analysis.cluster import Clusters, ClustersDict
from scipy.cluster import hierarchy as sp_hier


class Hierarchical(Base):
	def __init__(
			self,
			mice: Mice,
			key: str = None,
			mode: str = 'ca2',
			task: str = 'rest',
			method: str = 'ward',
			metric: str = 'correlation',
			**kwargs,
	):
		defaults = {
			'palette': 'tab20c',
			'leaf_font_size': 5,
			'no_labels': False,
			'orientation': 'top',
			'count_sort': 'ascending',
			'figsize': (13.5, 4),
		}
		super(Hierarchical, self).__init__(
			mice=mice,
			key=key,
			mode=mode,
			task=task,
			metric=metric,
			style_defaults=defaults,
			**kwargs,
		)
		self.method = method
		self.linkage = None
		self.results = {}
		self.clusters = {}

	def fit_hierarchical(
			self,
			n_clusters: Union[int, range, List[int]] = 0,
			c_idxs: dict = None, **kwargs, ):
		if isinstance(n_clusters, int):
			n_clusters = [n_clusters]
		c_idxs = c_idxs if c_idxs else {n: None for n in n_clusters}
		assert list(n_clusters) == list(c_idxs.keys())

		if self.linkage is None:
			self._fit_pdist()
		for k, v in self.style.items():
			if k not in kwargs:
				kwargs[k] = v

		for n in n_clusters:
			self.results[n] = get_clust_graph(
				linkage_mat=self.linkage,
				c_list=c_idxs[n],
				labels=list(self.l2n),
				num_clusters=n,
				**kwargs,
			)
		return self

	def make_cluster_masks(self):
		if not len(self.results):
			_ = self.fit_hierarchical()
		updated_results = {}
		for n, d in self.results.items():
			pass
			# community = self._cluster2community(d['cluster_assigns'])
			# cluster_masks = community2masks(community, self.mice)
			# updated_results[n] = {'cluster_masks': cluster_masks, **d}
		self.results = updated_results
		return self

	def make_clusters(self):
		if not len(self.results):
			_ = self.fit_hierarchical()
		_ = self.make_cluster_masks()
		for n, d in self.results.items():
			x3d, _ = masks2arr(
				masks=d['cluster_masks'],
				dims_flat=self.mice.al.dims_flat,
			)
			self.clusters[n] = Clusters(
				partition=x3d,
				m_struct=self.mice.parcel.results['major_masks'],
				c_cluster=d['cluster_colors'],
				c_struct=self.mice.parcel.results['major_colors'],
				threshold=0.0,
			)
		return self

	def load(self, file_hex: str, load_dir: str = None, make: bool = False):
		if load_dir is None:
			_dir = pjoin(self.mice.cfg.results_dir, 'hierarchical')
			matching_fits = [x for x in os.listdir(_dir) if self.metric in x]
			selected_fit = sorted(matching_fits)[-1]
			load_dir = pjoin(_dir, selected_fit)
		fits_dir = pjoin(load_dir, 'fits')
		path = pjoin(fits_dir, f"{file_hex}.npy")
		self.results = np.load(path, allow_pickle=True).item()
		if make:
			self.make_clusters()
		return self

	def _fit_pdist(self):
		hier_results = hierarchical_cluster(
			data=self.data,
			method=self.method,
			dist_metric=self.metric,
			link_metric=self.metric,
		)
		self.linkage = hier_results['z']
		return


def sort_lr(cluster: ClustersDict) -> ClustersDict:
	def sort_fn(x):
		if len(x.split('-')) == 2:
			return x[2:] + x[:2]
		else:
			return x.split('-')[0] + '{:05d}'.format(int(x.split('-')[-1]))

	cluster_sorted = {
		color: sorted(set(leaves), key=sort_fn)
		for color, leaves in cluster.items()
	}
	return ClustersDict(cluster_sorted)


def hierarchical_cluster(
		data: Union[dict, np.ndarray, pd.DataFrame],
		method: str = 'ward',
		dist_metric: str = 'correlation',
		link_metric: str = 'correlation', ):
	results = get_pdist(data, dist_metric)
	z = sp_hier.linkage(
		results['y'],
		method=method,
		metric=link_metric,
		optimal_ordering=True,
	)
	results.update({'z': z})
	return results


def get_cluster_classes(dn: dict, sort: bool = False):
	cluster_idxs = collections.defaultdict(set)
	accepted_legs = set()
	for c, pi in zip(dn['color_list'], dn['icoord']):
		for leg in pi[1:3]:
			if leg in accepted_legs:
				continue
			i = (leg - 5.0) / 10.0
			i_int = int(i)
			if abs(i - i_int) < 1e-5:
				cluster_idxs[c].add(i_int)
				accepted_legs.add(leg)

	cluster_classes = ClustersDict()
	for c, l in cluster_idxs.items():
		i_l = [dn['ivl'][i] for i in l]
		cluster_classes[c] = i_l
	cluster_classes = dict(sorted(cluster_classes.items()))
	cluster_classes = {rgb2hex(c): v for c, v in cluster_classes.items()}
	cluster_classes = ClustersDict(cluster_classes)

	if sort:
		cluster_classes = sort_lr(cluster_classes)

	return cluster_classes


def get_cluster_assigns(
		linkage_mat: np.ndarray,
		labels: List[str],
		num_clusters: int, ):

	cluster_list = sp_hier.fcluster(
		Z=linkage_mat,
		t=num_clusters,
		criterion='maxclust',
		monocrit=None,
	)
	cluster_assigns = collections.defaultdict(list)
	for leaf, cluster_idx in zip(labels, cluster_list):
		cluster_assigns[cluster_idx].append(leaf)
	cluster_assigns = dict(sorted(cluster_assigns.items()))
	return cluster_assigns


def get_cluster_colors(
		cluster_colors: Dict[int, Union[int, str]] = None,
		palette: str = 'tab20c', ):
	c_fixed = {}
	if all([isinstance(x, int) for x in cluster_colors.values()]):
		palette = sns.color_palette(palette, n_colors=20)
		for cluster_i, c_idx in cluster_colors.items():
			c_fixed[cluster_i] = rgb2hex(list(palette)[c_idx])
	else:
		c_fixed = {cluster_i: rgb2hex(c) for cluster_i, c in cluster_colors.items()}
	return c_fixed


def get_clust_graph(
		linkage_mat: np.ndarray,
		c_list: List[Union[int, str]] = None,
		palette: str = 'tab20c',
		labels: List[str] = None,
		num_clusters: int = 0,
		leaf_font_size: int = 5,
		no_labels: bool = False,
		orientation: str = 'top',
		count_sort: str = 'ascending',
		figsize=(13.5, 4), ):

	sns.set_style('whitegrid')
	fig, ax = plt.subplots(1, 1, figsize=figsize)

	dn = sp_hier.dendrogram(
		Z=linkage_mat,
		p=num_clusters,
		truncate_mode="lastp",
		color_threshold=None,
		get_leaves=True,
		orientation=orientation,
		labels=labels,
		count_sort=count_sort,
		distance_sort=False,
		show_leaf_counts=True,
		no_plot=False,
		no_labels=no_labels,
		leaf_font_size=leaf_font_size,
		leaf_rotation=-90 if orientation == 'top' else 0,
		leaf_label_func=None,
		show_contracted=True,
		link_color_func=None,
		ax=ax,
		above_threshold_color='C0',
	)
	fig.tight_layout()
	plt.close()

	cluster_classes = get_cluster_classes(dn, sort=True if num_clusters == 0 else False)
	if num_clusters == 0:
		cluster_assigns = {i+1: leaf_list for i, leaf_list in enumerate(cluster_classes.values())}
		c_list = list(cluster_classes.keys())
	else:
		cluster_assigns = get_cluster_assigns(linkage_mat, labels, num_clusters)
	c_list = c_list if c_list else [f"C{i}" for i in range(1, num_clusters + 1)]
	cluster_colors = {i + 1: c for i, c in enumerate(c_list)}
	cluster_colors = get_cluster_colors(cluster_colors, palette)

	output = {
		'cluster_classes': cluster_classes,
		'cluster_assigns': cluster_assigns,
		'cluster_colors': cluster_colors,
		'dendogram': dn,
		'fig': fig,
		'ax': ax,
	}
	return output


def reproducibility(
		mice: Mice,
		metric: str,
		n_clusters: Union[int, range, List[int]],
		load_dir: str,
		c_masks_dir: str,
		combos: List[list],
		pairs: List[tuple], ):

	hex2key = np.load(pjoin(load_dir, 'hex2key.npy'), allow_pickle=True).item()
	# noinspection PyUnresolvedReferences
	key2hex = {k: h for h, k in hex2key.items()}

	dict_list = []

	# Condition #1: between sessions
	condition = 'between-ses'
	pbar = tqdm(combos)
	for a in pbar:
		key_sub = f"sub-{','.join([str(x) for x in a])}"
		selected = {}
		for s in [[1], [2], [3]]:
			key_ses = f"ses-{','.join([str(x) for x in s])}"
			key = '_'.join([key_ses, key_sub])
			hex_f = key2hex[key]
			pbar.set_description(f"{key},\tfile name: {hex_f}")

			masks_file = pjoin(c_masks_dir, f"{hex_f}.npy")
			if os.path.isfile(masks_file):
				masks = np.load(
					pjoin(c_masks_dir, f"{hex_f}.npy"),
					allow_pickle=True,
				).item()
			else:
				hier = Hierarchical(
					mice=mice,
					metric=metric,
				).load(hex_f, load_dir).make_cluster_masks()
				masks = {
					n: d['cluster_masks']
					for n, d in hier.results.items()
				}
			selected[key_ses] = masks

		for k1, k2 in itertools.combinations(selected, 2):
			for n in n_clusters:
				m1 = selected[k1][n]
				m2 = selected[k2][n]
				scores, mappings = clusters_sim_scores(m1, m2, weighted=True)
				for k, v in scores.items():
					data_dict = {
						'condition': [condition],
						'sessions': ['_vs_'.join([k1, k2])],
						'n_rois': [mice.parcel.results['n_cortical_regions']],
						'n_clusters': [n],
						'metric': [metric],
						'dist_measure': [k],
						'val': [np.sum(np.diag(v))],
						'val_mean': [np.mean(np.diag(v))],
						'val_median': [np.median(np.diag(v))],
					}
					dict_list.append(data_dict)

	# Condition #2: within same session (between subjects)
	condition = 'between-sub'
	pbar = tqdm(pairs)
	for a, b in pbar:
		_key_a = f"sub-{','.join([str(x) for x in a])}"
		_key_b = f"sub-{','.join([str(x) for x in b])}"
		for s in [[1], [2], [3]]:
			key_ses = f"ses-{','.join([str(x) for x in s])}"
			key_a = '_'.join([key_ses, _key_a])
			key_b = '_'.join([key_ses, _key_b])
			hex_f_a = key2hex[key_a]
			hex_f_b = key2hex[key_b]
			pbar.set_description(f"{key_ses}  (a: {hex_f_a}, b: {hex_f_b})")

			masks_file = pjoin(c_masks_dir, f"{hex_f_a}.npy")
			if os.path.isfile(masks_file):
				m_a = np.load(
					masks_file,
					allow_pickle=True,
				).item()
			else:
				hier = Hierarchical(
					mice=mice,
					metric=metric,
				).load(hex_f_a, load_dir).make_cluster_masks()
				m_a = {
					n: d['cluster_masks']
					for n, d in hier.results.items()
				}
			masks_file = pjoin(c_masks_dir, f"{hex_f_b}.npy")
			if os.path.isfile(masks_file):
				m_b = np.load(
					masks_file,
					allow_pickle=True,
				).item()
			else:
				hier = Hierarchical(
					mice=mice,
					metric=metric,
				).load(hex_f_b, load_dir).make_cluster_masks()
				m_b = {
					n: d['cluster_masks']
					for n, d in hier.results.items()
				}

			for n in n_clusters:
				scores, mappings = clusters_sim_scores(m_a[n], m_b[n], weighted=True)
				for k, v in scores.items():
					data_dict = {
						'condition': [condition],
						'sessions': ['_vs_'.join([key_ses, key_ses])],
						'n_rois': [mice.parcel.results['n_cortical_regions']],
						'n_clusters': [n],
						'metric': [metric],
						'dist_measure': [k],
						'val': [np.sum(np.diag(v))],
						'val_mean': [np.mean(np.diag(v))],
						'val_median': [np.median(np.diag(v))],
					}
					dict_list.append(data_dict)

	# Get final results df and save
	results_df = pd.DataFrame(merge_dicts(dict_list))
	save_obj(
		obj=results_df,
		file_name='reproduc',
		save_dir=load_dir,
		mode='df',
		verbose=True,
	)
	return


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"n",
		help="number of ROIs",
		type=int,
	)
	parser.add_argument(
		"--n_clusters",
		help="max number of clusters",
		type=int,
		default=10,
	)
	parser.add_argument(
		"--mode",
		help="data modality, options: {'ca', 'bold'}",
		type=str,
		choices={'ca', 'bold'},
		default='bold',
	)
	parser.add_argument(
		"--metric",
		help="metric, choices: {'correlation', 'cosine', 'canberra', 'euclidean'}",
		type=str,
		choices={'correlation', 'cosine', 'canberra', 'euclidean'},
		default='correlation',
	)
	parser.add_argument(
		"--task",
		help="options: {'rest', 'led', 'both'}",
		type=str,
		choices={'rest', 'led', 'both'},
		default='rest',
	)
	parser.add_argument(
		"--save_masks",
		help="compute and save masks?",
		action="store_true",
		default=True,
	)
	return parser.parse_args()


def main():
	args = _setup_args()
	n_clusters = range(2, args.n_clusters)

	mice = Mice(Config(args.n))
	mice.combine_dfs(args.task)

	save_dir = pjoin(mice.cfg.results_dir, f"hierarchical")
	save_dir = pjoin(save_dir, f"{args.metric}-{now()}")
	c_masks_dir = pjoin(save_dir, 'cluster_masks')
	fits_dir = pjoin(save_dir, 'fits')
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(fits_dir, exist_ok=True)
	if args.save_masks:
		os.makedirs(c_masks_dir, exist_ok=True)

	import uuid
	import random
	rd = random.Random()
	rd.seed(mice.cfg.random_state)
	uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))

	def _get_hex_string(id_length=8):
		return uuid.uuid4().hex[:id_length]

	combos = []
	for a in itertools.combinations(mice.cfg.sub_ids, mice.cfg.num_subjects // 2):
		combos.append(list(a))
	pairs = [(combos[i], combos[-i - 1]) for i in range(len(combos) // 2)]
	assert all(set(x[0] + x[1]) == set(mice.cfg.sub_ids) for x in pairs)

	key2hex = {}
	for s in [[1], [2], [3], [1, 2, 3]]:
		for a in combos:
			key_ses = f"ses-{','.join([str(x) for x in s])}"
			key_sub = f"sub-{','.join([str(x) for x in a])}"
			key = '_'.join([key_ses, key_sub])
			key2hex[key] = _get_hex_string()
	for key in mice.bold_combined:
		key2hex[key] = _get_hex_string()

	hex2key = {h: k for k, h in key2hex.items()}
	save_obj(
		obj=hex2key,
		file_name='hex2key',
		save_dir=save_dir,
		mode='npy',
	)

	from tqdm import tqdm
	pbar = tqdm(key2hex.items(), position=0, dynamic_ncols=True, leave=True)
	for key, hex_f in pbar:
		pbar.set_description(f"{key}   (file: {hex_f})")

		if key in mice.bold_combined:
			df = mice.bold_combined[key]
		else:
			chosen_ses, chosen_sub = [
				[int(i) for i in x.split('-')[1].split(',')]
				for x in key.split('_')
			]
			df_list = []
			for sub in chosen_sub:
				for ses in chosen_ses:
					df_list.append(mice.bold_combined[mice.get_key(sub, ses)])
			df = pd.concat(df_list)

		# fit & save
		hier = Hierarchical(
			mice=mice,
			df=df,
			metric=args.metric,
		).fit(n_clusters)
		save_obj(
			obj=hier.results,
			file_name=hex_f,
			save_dir=fits_dir,
			mode='npy',
			verbose=False,
		)
		if args.save_masks:
			hier.make_cluster_masks()
			masks = {
				n: d['cluster_masks']
				for n, d in hier.results.items()
			}
			save_obj(
				obj=masks,
				file_name=hex_f,
				save_dir=c_masks_dir,
				mode='npy',
				verbose=False,
			)

	# run the reproduc analysis and save results df
	reproducibility(
		mice=mice,
		metric=args.metric,
		n_clusters=n_clusters,
		load_dir=save_dir,
		c_masks_dir=c_masks_dir,
		combos=combos,
		pairs=pairs,
	)

	print('\n')
	print(args)
	print("[PROGRESS] analysis.hierarchical done.\n")


if __name__ == "__main__":
	main()
