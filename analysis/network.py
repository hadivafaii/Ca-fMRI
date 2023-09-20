from analysis.base import *
from .network_helpers import make_random_model, save_edgelist

_ATTRS = [
	'adj',
	'graphs',
	'graphs_cfg',
	'graphs_cfg_cl',
	'centralities',
	'centralities_cfg',
	'centralities_cfg_cl',
]


class Network(Base):
	def __init__(
			self,
			binarize: bool = False,
			percentiles: Iterable[int] = None,
			cent_measures: List[str] = None,
			**kwargs,
	):
		style_defaults = {
			'l_colors': '#0C75C0',
			'r_colors': '#C0570C',
			'ticksize': 10,
			'ticks': True,
			'cmap': get_hm_cmap(),
			'fmt': 's',
			'vmin': 0.0,
			'vmax': 1.0,
			'linewidths': 0.01,
			'image_th': 0.0,
			'annot_th': 0.3,
			'annot_fontsize': 9,
			'annot_fontname': 'Loma',
			'cbar_aspect': 35,
			'cbar_shrink': 1.0,
			'tight_layout': False,
			'constrained_layout': True,
		}
		super(Network, self).__init__(
			style_defaults=style_defaults,
			**kwargs,
		)
		self._setup_centrality_fns()

		dense_compatible_metrics = [
			'pearson', 'spearman',
			'cosine', 'correlation',
		]
		if self.metric not in dense_compatible_metrics:
			print(f"Metric is '{self.metric}', must binarize")
			binarize = True
		self.binarize = binarize
		self.percentiles = None
		self.set_percentiles(
			percentiles=percentiles
			if percentiles is not None else
			np.linspace(35, 5, 7).astype(int),
		)
		self._cent_measures = cent_measures if cent_measures else [
			'degree',
			'subgraph',
			'closeness',
			'eigenvector',
			'betweenness',
			'clustering',
			'square_clustering',
			'average_neighbor_degree',
		]
		# variables to fit
		self.adj = None
		self.p_val = None
		self.thresholds = {}
		self.disconnected = {}
		self.adjacencies = {}
		self.graphs = {}
		self.graphs_cfg = {}
		self.graphs_cfg_cl = {}
		self.centralities = {}
		self.centralities_cfg = {}
		self.centralities_cfg_cl = {}

	def fit_network(
			self,
			n_samples: int = 10,
			force: bool = False,
			full: bool = False,
			save: bool = False, ):
		if not force:
			loaded = self.load_network(full)
			if loaded:
				return self

		# if not loaded or force; then fit
		if self.verbose:
			print('\n[PROGRESS] fit: initiated')

		bad_time = np.isnan(self.data).sum(0) == len(self.data)
		good_nodes = np.isnan(self.data).sum(1) == bad_time.sum()
		m = good_nodes.reshape(-1, 1) @ good_nodes.reshape(1, -1)
		x = self.data[good_nodes][:, ~bad_time]
		a = np_nans((len(self.data),) * 2)
		p = np_nans((len(self.data),) * 2)

		if self.metric in ['pearson', 'spearman']:
			_a, _p = get_adj_mat(x, self.metric, 1.0)
			a[m], p[m] = _a.flat, _p.flat
		else:
			output = get_pdist(x, self.metric)
			a[m] = self._dist2adj(output['y_sq_form']).flat

		self.adj, self.p_val = a, p

		self.style['vmax'] = np.nanmax(self.adj)
		self.apply_thresholds()
		self.make_graphs(full)
		if save:
			self.save_network()
			self.save_graphs(self.mice.cfg.svinet_dir, n_samples)
		if self.verbose:
			print('[PROGRESS] fit: done!\n\n')
		return self

	def save_graphs(self, path: str, n_samples: int):
		path = pjoin(path, self.base_name)
		self._sv_g_parallel(path, n_samples, 'real', self.graphs)
		self._sv_g_parallel(path, n_samples, 'config', self.graphs_cfg)
		self._sv_g_parallel(path, n_samples, 'config_clust', self.graphs_cfg_cl)

	def _sv_g_parallel(self, path, n_samples, tag, graphs_dict, n_jobs=-1):
		kwargs = {
			'save_dir': pjoin(path, tag),
			'n_samples': n_samples,
			'random_state': self.mice.cfg.random_state,
			'verbose': False,
		}
		with joblib.parallel_backend('multiprocessing'):
			joblib.Parallel(n_jobs=n_jobs)(
				joblib.delayed(save_graph)(graph, p, **kwargs)
				for p, graph in graphs_dict.items()
			)

		# with joblib.parallel_backend('threading'):
		# joblib.Parallel(n_jobs=n_jobs)(
		# 		joblib.delayed(_sv)(graph, p)
	# 			for p, graph in graphs_dict.items()
	# 		)
		return

	def save_network(self):
		save_dir = self.path(True)
		for attr in _ATTRS:
			save_obj(
				obj=getattr(self, attr),
				file_name=attr,
				save_dir=save_dir,
				mode='npy',
				verbose=self.verbose,
			)

	def load_network(self, full: bool = False):
		load_dir = [
			self.mice.cfg.results_dir,
			'Network',
			self.base_name,
		]
		load_dir = pjoin(*load_dir)

		try:
			self.adj = np.load(pjoin(load_dir, 'adj.npy'))
		except FileNotFoundError:
			return False

		if not full:
			return self

		for attr in _ATTRS:
			if attr == 'adj':
				continue
			file = pjoin(load_dir, f"{attr}.npy")
			v = np.load(file, allow_pickle=True).item()
			setattr(self, attr, v)

		self.set_percentiles(list(self.graphs))
		self.apply_thresholds()
		return self

	def set_percentiles(self, percentiles, fit: bool = False):
		percentiles = [int(x % 100) for x in percentiles]
		percentiles = list(set(percentiles))
		self.percentiles = sorted(percentiles, reverse=True)
		if fit:
			return self.fit_network()
		return

	def _dist2adj(self, dist):
		np.fill_diagonal(dist, 1.)
		if self.metric in ['cosine', 'correlation', 'braycurtis']:
			adj = 1 - dist
		else:
			adj = np.min(dist) / dist
		np.fill_diagonal(adj, 0.)
		return adj

	def apply_thresholds(self):
		rho = [p / 100 for p in self.percentiles]
		thresholds = calculate_threshold(self.adj, rho)
		thresholds = np.maximum(thresholds, 0.0)
		self.thresholds = dict(zip(self.percentiles, thresholds))
		for p, th in zip(self.percentiles, thresholds):
			a = self.adj.copy()
			a[a <= th] = 0.
			if self.binarize:
				a[a > th] = 1.
			zero = np.nanmean(a, 0) == 0
			self.disconnected[p] = np.where(zero)[0]
			nonzero = np.nanmean(a, 0) > 0
			m = nonzero.reshape(-1, 1) @ nonzero.reshape(1, -1)
			a[~m] = np.nan
			self.adjacencies[p] = a

	def make_graphs(self, full: bool):
		for name, a in self.adjacencies.items():
			g = self._adj2graph(a)
			self.graphs[name] = g
			if full:
				self.graphs_cfg[name] = make_random_model(
					g=g, clustered=False,
					weighted=not self.binarize,
					random_state=self.mice.cfg.random_state)
				self.graphs_cfg_cl[name] = make_random_model(
					g=g, clustered=True,
					weighted=not self.binarize,
					random_state=self.mice.cfg.random_state)
		if self.verbose:
			print('[PROGRESS] graphs: done')

	def compute_centralities(self):
		for p, g in self.graphs.items():
			self.centralities[p] = self._centrality_df(
				g=g, name=p)
			self.centralities_cfg[p] = self._centrality_df(
				g=self.graphs_cfg[p], name=p)
			self.centralities_cfg_cl[p] = self._centrality_df(
				g=self.graphs_cfg_cl[p], name=p)
		if self.verbose:
			print('[PROGRESS] centralities: done')
		return

	def _centrality_df(self, g, name):
		dict_list = []

		# weighted degree
		dw = dict(g.degree(weight='weight'))
		dw = _normalize_vals(dw)
		data_dict = {
			'p': [name] * g.order(),
			'measure': ['degree_w'] * g.order(),
			'node': dw.keys(),
			'centrality': dw.values(),
		}
		dict_list.append(data_dict)

		# typical centrality measures
		for lbl, f in self._centrality_fn.items():
			if lbl not in self._cent_measures:
				continue

			params = inspect.signature(f).parameters
			if 'k' in params:
				c = _normalize_vals(f(g, k=g.order()))
			else:
				c = _normalize_vals(f(g))
			data_dict = {
				'p': [name] * g.order(),
				'measure': [lbl] * g.order(),
				'node': c.keys(),
				'centrality': c.values(),
			}
			dict_list.append(data_dict)

			if 'weight' in params and not self.binarize:
				if 'k' in params:
					c = f(g, weight='weight', k=g.order())
				else:
					c = f(g, weight='weight')
				c = _normalize_vals(c)
				data_dict = {
					'p': [name] * g.order(),
					'measure': [f"{lbl}_w"] * g.order(),
					'node': c.keys(),
					'centrality': c.values(),
				}
				dict_list.append(data_dict)

		# other centrality measures
		for lbl, f in self._other_centrality_fn.items():
			if lbl not in self._cent_measures:
				continue

			if lbl == 'hits':
				h, a = tuple(map(_normalize_vals, f(g)))
				data_dict = {
					'p': [name] * g.order(),
					'measure': ['hub'] * g.order(),
					'node': h.keys(),
					'centrality': h.values(),
				}
				dict_list.append(data_dict)
				data_dict = {
					'p': [name] * g.order(),
					'measure': ['authority'] * g.order(),
					'node': a.keys(),
					'centrality': a.values(),
				}
				dict_list.append(data_dict)

			elif lbl == 'voterank':
				v = f(g)
				v = {
					n: 1 - v.index(n) / len(v)
					if n in v else 0.
					for n in g
				}
				v = _normalize_vals(v)
				data_dict = {
					'p': [name] * g.order(),
					'measure': [lbl] * g.order(),
					'node': v.keys(),
					'centrality': v.values(),
				}
				dict_list.append(data_dict)

			elif lbl == 'dispersion':
				dispersion_mat = np.zeros((g.order(), g.order()))
				for i, d in f(g).items():
					for j, val in d.items():
						dispersion_mat[i, j] = val
				d_col = dispersion_mat.mean(0)
				d_row = dispersion_mat.mean(1)
				d_col = {n: d_col[n] for n in range(g.order())}
				d_row = {n: d_row[n] for n in range(g.order())}
				d_col, d_row = tuple(map(_normalize_vals, [d_col, d_row]))
				data_dict = {
					'p': [name] * g.order(),
					'measure': [f"{lbl}_col"] * g.order(),
					'node': d_col.keys(),
					'centrality': d_col.values(),
				}
				dict_list.append(data_dict)
				data_dict = {
					'p': [name] * g.order(),
					'measure': [f"{lbl}_row"] * g.order(),
					'node': d_row.keys(),
					'centrality': d_row.values(),
				}
				dict_list.append(data_dict)

			elif lbl == 'local_reaching':
				lr = {}
				lr_w = {}
				for n in g:
					lr[n] = f(g, n)
					if not self.binarize:
						lr_w[n] = f(g, n, weight='weight')
				lr, lr_w = tuple(map(_normalize_vals, [lr, lr_w]))
				data_dict = {
					'p': [name] * g.order(),
					'measure': [lbl] * g.order(),
					'node': lr.keys(),
					'centrality': lr.values(),
				}
				dict_list.append(data_dict)
				if not self.binarize:
					data_dict = {
						'p': [name] * g.order(),
						'measure': [f"{lbl}_w"] * g.order(),
						'node': lr_w.keys(),
						'centrality': lr_w.values(),
					}
					dict_list.append(data_dict)

			else:
				raise NotImplementedError

		df = pd.DataFrame(merge_dicts(dict_list))
		return df

	def _setup_centrality_fns(self):
		self._centrality_fn = {
			'degree': nx.degree_centrality,
			'harmonic': nx.harmonic_centrality,
			'closeness': nx.closeness_centrality,
			'betweenness': nx.betweenness_centrality,
			'eigenvector': nx.eigenvector_centrality_numpy,
			'average_neighbor_degree': nx.average_neighbor_degree,
			'current_flow_closeness': nx.current_flow_closeness_centrality,
			'current_flow_betweenness': nx.current_flow_betweenness_centrality,
			'communicability_betweenness': nx.communicability_betweenness_centrality,
			'information': nx.information_centrality,
			'percolation': nx.percolation_centrality,
			'clustering': nx.clustering,
			'square_clustering': nx.square_clustering,
			'subgraph': nx.subgraph_centrality,
			'subgraph_exp': nx.subgraph_centrality_exp,
			'second_order': nx.second_order_centrality,
			'load': nx.load_centrality,
		}
		self._group_centrality_fn = {
			'group_degree': nx.group_degree_centrality,
			'group_closeness': nx.group_closeness_centrality,
			'group_betweenness': nx.group_betweenness_centrality,
		}
		self._other_centrality_fn = {
			'hits': nx.hits,
			'voterank': nx.voterank,
			'dispersion': nx.dispersion,
			'local_reaching': nx.local_reaching_centrality,
		}

	def _adj2graph(self, adj):
		a = adj.copy()
		a[np.isnan(a)] = 0
		g = nx.from_numpy_array(a)
		if self.binarize:
			for n1, n2, d in g.edges(data=True):
				d.pop('weight', None)
		return g

	def _get_style_defaults(self):
		return

	def get_hemisphere_adj(self, a: Union[int, np.ndarray] = None):
		if isinstance(a, int):
			a = self.adjacencies[a]
		elif a is None:
			a = self.adj
		assert all(i == len(self.n2id) for i in a.shape)

		dim = len(self.n2id) // 2
		lower = np.tril_indices(dim)
		upper = np.triu_indices(dim)
		left = a[self.l_idxs][:, self.l_idxs]
		right = a[self.r_idxs][:, self.r_idxs]

		within_hemis = np.zeros((dim, dim))
		within_hemis[lower] = left[lower]
		within_hemis[upper] = right[upper]
		np.fill_diagonal(within_hemis, np.nan)
		between_hemis = a[self.l_idxs][:, self.r_idxs]
		return within_hemis, between_hemis

	def _get_fig_name(self, condition, name):
		file = [
			self.base_name,
			condition,
		]
		if name:
			file += [f"{name}"]
		return '_'.join(file) + '.pdf'

	def fig_within_hemis(
			self,
			a: Union[int, np.ndarray] = None,
			save_dir: str = None,
			name: str = None,
			**kwargs, ):
		for k, v in self.style.items():
			if k not in kwargs:
				kwargs[k] = v
		plt_defaults = {
			'figsize': (13, 12) if self.mice.cfg.nn == 64 else (20, 18),
			'square': False,
		}
		for k, v in plt_defaults.items():
			if k not in kwargs:
				kwargs[k] = v

		x, _ = self.get_hemisphere_adj(a)
		x[x < kwargs['image_th']] = 0.

		fig, ax_l = create_figure(
			nrows=1,
			ncols=1,
			figsize=kwargs['figsize'],
			tight_layout=kwargs['tight_layout'],
			constrained_layout=kwargs['constrained_layout'],
		)
		sns.heatmap(
			data=x,
			cmap=kwargs['cmap'],
			vmin=kwargs['vmin'],
			vmax=kwargs['vmax'],
			square=kwargs['square'],
			fmt=kwargs['fmt'],
			linewidths=kwargs['linewidths'],
			annot=_get_annot_str(x, kwargs['annot_th']),
			annot_kws={
				'fontsize': kwargs['annot_fontsize'],
				'fontname': kwargs['annot_fontname']},
			cbar_kws={
				'aspect': kwargs['cbar_aspect'],
				'shrink': kwargs['cbar_shrink']},
			ax=ax_l,
		)
		if kwargs['ticks']:
			ax_l.tick_params(
				axis='x',
				bottom=True,
				top=True,
				labelbottom=True,
				labeltop=False,
				rotation=90,
				colors=kwargs['l_colors'],
				labelsize=kwargs['ticksize'],
			)
			ax_l.tick_params(
				axis='y',
				left=True,
				right=True,
				labelleft=True,
				labelright=False,
				rotation=0,
				colors=kwargs['l_colors'],
				labelsize=kwargs['ticksize'],
			)
			ax_l.set_xticks(np.arange(0.5, len(x), 1))
			ax_l.set_yticks(np.arange(0.5, len(x), 1))
			# TODO: fix ticks
			# ax_l.set_xticklabels(self.l_lbls)
			# ax_l.set_yticklabels(self.l_lbls)

			ax_r = ax_l.twinx()
			ax_t = ax_l.twiny()
			for ax in [ax_r, ax_t]:
				ax.set_xlim((0, x.shape[1]))
				ax.set_ylim((x.shape[0], 0))
				ax.set_xticks(ax_l.get_xticks())
				ax.set_yticks(ax_l.get_yticks())
			# TODO: fix ticks
			# ax_r.set_yticklabels(self.r_lbls)
			# ax_t.set_xticklabels(self.r_lbls)
			ax_r.tick_params(
				axis='y',
				rotation=0,
				colors=kwargs['r_colors'],
				labelsize=kwargs['ticksize'],
			)
			ax_t.tick_params(
				axis='x',
				rotation=90,
				colors=kwargs['r_colors'],
				labelsize=kwargs['ticksize'],
			)
			axes = (ax_l, ax_r, ax_t)
		else:
			remove_ticks(ax_l)
			axes = (ax_l,)

		if save_dir is not None:
			os.makedirs(save_dir, exist_ok=True)
			if isinstance(a, int) and name is None:
				name = f"adj-p{a}"
			file = self._get_fig_name('within_hemis', name)
			file = pjoin(save_dir, file)
			fig.savefig(file, bbox_inches='tight', dpi=600)
			plt.close()
		return fig, axes

	def fig_between_hemis(
			self,
			a: Union[int, np.ndarray] = None,
			save_dir: str = None,
			name: str = None,
			**kwargs, ):
		for k, v in self.style.items():
			if k not in kwargs:
				kwargs[k] = v
		plt_defaults = {
			'figsize': (16, 12) if self.mice.cfg.nn == 64 else (20, 17),
			'square': True,
		}
		for k, v in plt_defaults.items():
			if k not in kwargs:
				kwargs[k] = v

		_, x = self.get_hemisphere_adj(a)
		x[x < kwargs['image_th']] = 0.

		fig, ax = create_figure(
			nrows=1,
			ncols=1,
			figsize=kwargs['figsize'],
			tight_layout=kwargs['tight_layout'],
			constrained_layout=kwargs['constrained_layout'],
		)
		sns.heatmap(
			data=x,
			cmap=kwargs['cmap'],
			vmin=kwargs['vmin'],
			vmax=kwargs['vmax'],
			square=kwargs['square'],
			fmt=kwargs['fmt'],
			linewidths=kwargs['linewidths'],
			annot=_get_annot_str(x, kwargs['annot_th']),
			annot_kws={
				'fontsize': kwargs['annot_fontsize'],
				'fontname': kwargs['annot_fontname']},
			cbar_kws={
				'aspect': kwargs['cbar_aspect'],
				'shrink': kwargs['cbar_shrink']},
			ax=ax,
		)
		if kwargs['ticks']:
			ax.tick_params(
				axis='x',
				bottom=False,
				top=True,
				labelbottom=False,
				labeltop=True,
				rotation=-90,
				colors=kwargs['r_colors'],
				labelsize=kwargs['ticksize'],
			)
			ax.tick_params(
				axis='y',
				left=True,
				right=False,
				labelleft=True,
				labelright=False,
				rotation=0,
				colors=kwargs['l_colors'],
				labelsize=kwargs['ticksize'],
			)
			ax.set_xticks(np.arange(0.5, len(x), 1))
			ax.set_yticks(np.arange(0.5, len(x), 1))
			# TODO: fix ticks
			# ax.set_xticklabels(self.r_lbls)
			# ax.set_yticklabels(self.l_lbls)
		else:
			remove_ticks(ax)

		if save_dir is not None:
			os.makedirs(save_dir, exist_ok=True)
			file = self._get_fig_name('between_hemis', name)
			file = pjoin(save_dir, file)
			fig.savefig(file, bbox_inches='tight', dpi=600)
			plt.close()
		return fig, ax


def collect_degrees_lite(
		mice: Mice,
		mode: str,
		rho: float,
		filter_isolated: bool = False, ):
	proc = mice.get_data_containers(
		mode=mode, force=True)[0]
	degrees = {}
	for key, f in proc.items():
		if mice.looper[key][-1] == -1:
			continue
		net = Network(
			mice=mice,
			key=key,
			mode=mode,
			task=f.task,
		).load_network(full=False)
		if not net:
			continue
		a = net.adj.copy()
		good = np.isfinite(a).sum(0) > 0
		thres = calculate_threshold(a, rho).pop()
		a[net.adj <= thres] = 0
		a[net.adj > thres] = 1
		deg = bn.nansum(a, -1)
		deg[~good] = np.nan
		if filter_isolated:
			deg[deg == 0] = np.nan
		degrees[key] = deg
	return degrees


def collect_degrees(
		mice: Mice,
		mode: str,
		rho: Union[float, Iterable[float]] = None, ):
	adj, dirs = collect_adjacencies(mice, mode)
	rho = rho if rho else np.linspace(0.05, 0.30, 6)
	if not isinstance(rho, Iterable):
		rho = [rho]
	adj_bin = collections.defaultdict(list)
	for i, a in enumerate(adj):
		thresholds = calculate_threshold(a, rho)
		for d, th in zip(rho, thresholds):
			_a = a.copy()
			_a[a <= th] = 0
			_a[a > th] = 1
			adj_bin[d].append(np.expand_dims(_a, 0))
	adj_bin = {k: np.concatenate(v) for k, v in adj_bin.items()}
	deg = {k: bn.nansum(v, -1) for k, v in adj_bin.items()}
	deg_avg = {k: bn.nanmean(v, 0) for k, v in deg.items()}
	output = {
		'keys': list(dirs),
		'adj': adj,
		'deg': deg,
		'adj_bin': adj_bin,
		'deg_avg': deg_avg,
	}
	return output


def collect_adjacencies(
		mice: Mice,
		mode: str,
		fill: float = None, ):
	proc = mice.get_data_containers(
		mode=mode, force=True)[0]
	adjacencies, dirs = [], {}
	for key, f in proc.items():
		if mice.looper[key][-1] == -1:
			continue
		net = Network(
			mice=mice,
			key=key,
			mode=mode,
			task=f.task,
		).load_network(full=False)
		if not net:
			continue
		adjacencies.append(np.expand_dims(
			net.adj, 0))
		dirs[key] = net.path(False)
	adjacencies = np.concatenate(adjacencies)
	if fill is not None:
		adjacencies[
			:,
			np.arange(adjacencies.shape[1]),
			np.arange(adjacencies.shape[2]),
		] = float(fill)
	return adjacencies, dirs


def collect_graphs(
		mice: Mice,
		mode: str,
		perc: int = 10,
		**kwargs, ):
	task = kwargs.get('task')
	if task is None:
		task = 'rest'
		kwargs['task'] = task
	proc, *_ = mice.get_data_containers(
		mode=mode, force=True, **kwargs)

	graphs = {}
	for key in proc:
		if mice.looper[key][-1] == -1:
			continue
		net = Network(
			mice=mice,
			key=key,
			mode=mode,
			task=task,
		).load_network(full=True)
		if not net:
			continue
		graphs[key] = net.graphs[perc]
	return graphs


def save_graph(
		graph: nx.Graph,
		perc: int,
		save_dir: str,
		n_samples: int,
		random_state: int = 42,
		verbose: bool = False, ):
	# save binarized
	name = f"p{perc}-binary"
	sv_dir = pjoin(save_dir, name)
	os.makedirs(sv_dir, exist_ok=True)
	save_edgelist(
		graph=graph,
		file_name=f"{name}.txt",
		save_dir=sv_dir,
		weighted=False,
		verbose=verbose,
	)
	# save sampled
	name = f"p{perc}-sample"
	sv_dir = pjoin(save_dir, name)
	os.makedirs(sv_dir, exist_ok=True)
	save_edgelist(
		graph=graph,
		file_name=f"{name}.txt",
		save_dir=sv_dir,
		weighted=True,
		sampler='binomial',
		sampler_kws={'n': n_samples, 'adj': 'p'},
		random_state=random_state,
		verbose=verbose,
	)
	return


def _get_annot_str(x, th):
	annot = []
	for row in x:
		annot.append([
			f"{e:.1f}" if abs(e) > th else ''
			for e in row
		])
	return np.array(annot)


def _normalize_vals(x: dict):
	try:
		return {
			k: v / max(list(x.values()))
			for k, v in x.items()
		}
	except ValueError:
		return {}


def _setup_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"nn",
		help='number of ROIs',
		type=int,
	)
	parser.add_argument(
		"ll",
		help='number of ROI layers',
		type=int,
	)
	parser.add_argument(
		"--resolution",
		help='Allen CCF resolution',
		type=int,
		default=100,
	)
	parser.add_argument(
		"--parcel_mode",
		help='parcellation mode',
		type=str,
		default='columnar',
		choices={'columnar', 'spatial', 'allen'},
	)
	parser.add_argument(
		"--mode",
		help="data modality, options: {'ca2', 'bold'}",
		type=str,
		choices={'ca2', 'bold'},
		default='bold',
	)
	parser.add_argument(
		"--desc",
		help="data description",
		type=str,
		default='all',
	)
	parser.add_argument(
		"--task",
		help="options: {'rest', 'led', 'all'}",
		type=str,
		choices={'rest', 'led', 'all'},
		default='all',
	)
	parser.add_argument(
		"--metric",
		help='metric to build network',
		type=str,
		choices={'pearson', 'spearman', 'cosine', 'canberra'},
		default='pearson',
	)
	parser.add_argument(
		"--key",
		help='key to choose data from Mice, default: loop over all',
		type=str,
		default='all',
	)
	parser.add_argument(
		"--full",
		help='fit config and config clustered as well?',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--binarize",
		help='binarize the adjacency matrix?',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--verbose",
		help='prints progress',
		action='store_true',
		default=False,
	)
	return parser.parse_args()


def main():
	args = _setup_args()
	mice = Mice(Config(
		nn=args.nn,
		ll=args.ll,
		resolution=args.resolution,
		parcel_mode=args.parcel_mode,
	))
	percentiles = np.linspace(30, 5, 6).astype(int)
	props = {
		'mice': mice,
		'mode': args.mode,
		'task': args.task,
		'metric': args.metric,
		'binarize': args.binarize,
		'percentiles': percentiles,
		'verbose': args.verbose,
		'prep_data': True,
	}
	if args.mode == 'ca2':
		# TODO: remove this later
		# bands = [0.01, 0.5, (0.01, 0.5)]
		bands = [0.5, (0.01, 0.5)]
	else:
		bands = [0.01]
	if args.task == 'all':
		tasks = ['rest', 'led']
	else:
		tasks = [args.task]
	if args.desc == 'all':
		descs = default_descs(args.mode)
	else:
		descs = [args.desc]
	pbar1 = tqdm(
		itertools.product(tasks, descs, bands),
		position=0,
		leave=True,
		dynamic_ncols=True,
		total=len(tasks)*len(descs)*len(bands),
	)
	for task, desc, b in pbar1:
		kws = {
			'task': task,
			f"band_{args.mode.split('-')[0]}": b,
			f"desc_{args.mode.split('-')[0]}": desc,
		}
		mice.setup_func_data(**kws)
		if args.key == 'all':
			pbar2 = tqdm(
				mice.get_data_containers(args.mode)[0],
				position=-1,
				leave=False,
				dynamic_ncols=True,
			)
			for key in pbar2:
				msg = f"running net (task-{task}, desc-{desc}, "
				msg += f"b-{str(b).replace(' ', '')})"
				msg += f": {key}"
				pbar1.set_description(msg)
				_ = Network(key=key, **props).fit_network(
					force=False, full=args.full, save=True)
		else:
			_ = Network(key=args.key, **props).fit_network(
				force=False, full=args.full, save=True)

	print('\n')
	print(args)
	print(f"[PROGRESS] analysis.network done {now(True)}.\n")


if __name__ == "__main__":
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		main()
