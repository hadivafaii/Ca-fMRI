from analysis.network import *
from analysis.cluster import Clusters
from prettytable import PrettyTable

_ATTRS = [
	'gam', 'lam', 'pi', 'perms',
	'gam_all', 'lam_all', 'pi_all',
]


class SVINET(Network):  # TODO: SVINET should be Base, not Network
	def __init__(
			self,
			perc: str = 'p20-sample',
			match_metric: str = 'euclidean',
			match_using: str = 'gam',
			graph_type: str = 'real',
			kws_load: List[str] = None,
			normalize: bool = True,
			**kwargs,
	):
		super(SVINET, self).__init__(**kwargs)
		self.perc = perc
		self.graph_type = graph_type
		self.match_using = match_using
		self.match_metric = match_metric
		self.normalize = normalize

		kws_load = kws_load if kws_load else 'linksampling'
		if not isinstance(kws_load, list):
			self.kws_load = [kws_load]
		self.filter_fn = _get_filter_fn(self.num_k, *self.kws_load)
		self._runs = collections.defaultdict(dict)
		self.results = collections.defaultdict(dict)

		# results to fit
		self.pi_cut_df = pd.DataFrame()
		self.best_qs = []
		self.clusters = None
		self.centroids = None
		self.perms = None

		# for plotting purposes  # TODO: these are irrelevant now
		self.pi2d = None
		self.overlap2d = None
		self.disjoint2d = None
		self.contour2d = None

	def fit_svinet(self, force: bool = False, **kwargs):
		if not force:
			loaded = self.load_results()
			if loaded is not None:
				return self
			else:
				if self.verbose:
					print('Fit not found . . . initiating fit')

		if not len(self._runs):
			self.load_runs()
		if not len(self._runs):
			if self.verbose:
				print('Warning, runs not loaded')
			return self

		# fit
		if self.verbose:
			print('\n[PROGRESS] fit: initiated')
		data = self._cat_data(
			raw=True,
			which=self.match_using,
		)
		centroids, _ = fit_kmeans(
			data=data,
			n_clusters=self.num_k,
			match_metric=self.match_metric,
			random_state=self.mice.cfg.random_state,
			normalize=self.normalize,
			verbose=self.verbose,
			**kwargs,
		)
		global_mapping = self.align_centroid_to_structs(
			pi_mv=centroids,
			metric='cosine',
			cortex_only=True,
			global_order=False,
			assert_equal=False,
		)
		self.centroids = centroids[global_mapping]
		self.perms = self._perms(data, 'correlation')
		self._align()
		if self.verbose:
			print('[PROGRESS] align: done')
		# process & save
		self._calculate_pi_gam_lam()
		self._save()
		# self.process_results(self.plot_cortex_only)  # TODO: fix this function
		if self.verbose:
			print('[PROGRESS] fit: done!\n\n')
		return self

	def _save(self):
		for name in _ATTRS:
			save_obj(
				obj=getattr(self, name),
				file_name=name,
				save_dir=self.path(True),
				verbose=self.verbose,
				mode='npy',
			)

	def save_plots(self):  # TODO: not needed anymore
		_ = self.plot_memberships('bar', show=False, save=True)
		_ = self.plot_memberships('box', show=False, save=True)
		_ = self.plot_memberships('box-strip', show=False, save=True)
		_ = self.plot_communities_2d(show=False, disjoint=True)
		_ = self.plot_communities_2d(show=False, disjoint=False)
		_ = self.plot_membership_heatmap(show=False)
		_ = self.plot_swarm('degree', show=False)
		_ = self.plot_swarm('entropy', show=False)
		_ = self.plot_4block_percentages(show=False)
		_ = self.plot_js_vs_weight(show=False)
		_ = self.plot_snr(sort=False, show=False)
		_ = self.plot_snr(sort=True, show=False)
		self.make_centrality_plots()
		self.render()
		print('[PROGRESS] save plots & render: done')
		return

	def name(self):
		name = [
			self.graph_type,
			f"k{self.num_k}",
			self.perc,
			self.match_metric,
			*self.kws_load,
			f"({now()})",
		]
		name = [e for e in name if e]
		name = '-'.join(name)
		return name

	def path(self, mkdir: bool = False):
		path = [
			self.mice.cfg.results_dir,
			type(self).__name__,
			self.base_name,
			self.name(),
		]
		path = pjoin(*path)
		if mkdir:
			os.makedirs(path, exist_ok=True)
		return path

	def _calculate_pi_gam_lam(self):
		lam_all = self._cat_data(raw=False, which='lam')
		gam_all = self._cat_data(raw=False, which='gam')
		nonzero = gam_all.sum(-2, keepdims=True) != 0
		original_shape = gam_all.shape
		ner = Normalizer()
		gam_all = ner.fit_transform(
			gam_all.reshape(original_shape[0], -1))
		gam_all = gam_all.reshape(original_shape)
		gam_all *= self.num_k * original_shape[-1]
		gam_all = np.where(nonzero, gam_all, np.nan)
		pi_all = gam_all / np.nansum(gam_all, -2, keepdims=True)

		self.lam_all = lam_all
		self.gam_all = gam_all
		self.pi_all = pi_all
		self.lam = np.nanmean(self.lam_all, 0)
		self.gam = np.nanmean(self.gam_all, 0)
		self.pi = self.gam / np.nansum(self.gam, 0, keepdims=True)
		return

	def process_results(self):
		self._calculate_entropies()
		self._calculate_bridgeness()
		self.make_clusters()
		if self.verbose:
			print('[PROGRESS] process results: done')
		return

	def _calculate_entropies(self):
		a, b = np.where(self.pi_all.sum(1) == 0)
		self.pi_all[a, :, b] = np.nan
		self.pi[:, np.where(self.pi.sum(0) == 0)[0]] = np.nan
		self.h_all = sp_stats.entropy(
			self.pi_all, axis=1) / np.log(self.num_k)
		self.h = sp_stats.entropy(self.pi) / np.log(self.num_k)
		return

	def _calculate_bridgeness(self):
		b_all = self.pi_all - np.array([1 / self.num_k] * self.num_k).reshape((1, -1, 1))
		b_all = np.linalg.norm(b_all, ord=2, axis=1) ** 2
		b_all *= self.num_k / (self.num_k + 1)
		self.b_all = 1 - np.sqrt(b_all)

		b = self.pi - np.array([1 / self.num_k] * self.num_k).reshape(-1, 1)
		b = np.linalg.norm(b, ord=2, axis=0) ** 2
		b *= self.num_k / (self.num_k + 1)
		self.b = 1 - np.sqrt(b)
		return

	def get_result_dict(self, p: int = None):
		p = p if p else self.perc
		return self.results[p]

	def get_membership_df(self, num_cuts: int = 10, decimals: int = 2):
		bins = np.linspace(0, 1, num_cuts + 1)

		dict_list = []
		if self.verbose:
			pbar = tqdm(self.get_result_dict().items())
		else:
			pbar = self.get_result_dict().items()
		for run_name, run in pbar:
			if self.verbose:
				msg = f"creating pi df, run: {run_name}"
				pbar.set_description(msg)
			for i in range(self.num_k):
				mem = np.round(run['pi'][i], decimals=decimals)
				mem_cut = pd.cut(mem[mem > 0], bins).value_counts()
				counts = mem_cut.values.tolist()
				if not sum(counts):
					if self.verbose:
						print('Warning, type 3', run_name, i)
					continue
				percent = [x / sum(counts) * 100 for x in counts]
				interval = mem_cut.index.tolist()
				data_dict = {
					'run': [run_name] * len(mem_cut),
					'oc_idx': [i + 1] * len(mem_cut),
					'count': counts,
					'percent': percent,
					'interval': interval,
				}
				dict_list.append(data_dict)
		pi_cut_df = pd.DataFrame(merge_dicts(dict_list))
		return pi_cut_df

	def _disjoint_from_pi(self, pi):
		community = self.get_disjoint_community(pi)
		masks = self.community2masks(community)
		for idx in range(1, self.num_k + 1):
			if idx not in masks:
				masks[idx] = np.zeros(self.mice.al.dims, dtype=bool)
		masks = dict(sorted(masks.items()))
		return masks

	def load_results(
			self,
			perc: str = None,
			fit_id: int = -1, ):
		if perc is not None:
			self.perc = perc
		load_dir = [
			self.mice.cfg.results_dir,
			self.__class__.__name__,
			self.base_name,
		]
		load_dir = pjoin(*load_dir)
		try:
			matches = [
				str(x) for x in
				os.listdir(load_dir)
			]
		except FileNotFoundError:
			if self.verbose:
				print('Warning, fits dir is empty')
			return
		patterns = [
			self.graph_type,
			f"k{self.num_k}",
			self.perc,
			self.match_metric,
		]
		matches = [
			x for x in sorted(matches, key=sort_date_latest)
			if all(k in x for k in patterns)
		]
		if not len(matches):
			if self.verbose:
				print('Warning, no matching fits found')
			return
		if self.verbose:
			msg = f"[PROGRESS] loading '{matches[fit_id]}'"
			msg += f" from\n'{load_dir}'"
			print(msg)

		path = pjoin(load_dir, matches[fit_id])

		for name in _ATTRS:
			if name == 'perms':
				continue
			file_npy = pjoin(path, f"{name}.npy")
			setattr(self, name, np.load(file_npy, mmap_mode='r'))
		if self.verbose:
			print(f"[PROGRESS] results loaded")
		return self

	def load_runs(
			self,
			runs_path: str = None,
			graph_type: str = None,
			perc: str = None, ):
		if perc:
			self.perc = perc
		if self.perc == 'all':
			self.load_network()
			p_loop = [
				f"p{p}-sample" for p
				in self.percentiles
			]
		else:
			p_loop = [self.perc]

		_path = runs_path if runs_path else self.mice.cfg.svinet_dir
		_type = graph_type if graph_type else self.graph_type
		for p in p_loop:
			path = pjoin(_path, self.base_name, _type, p)
			runs = next(os.walk(path))[1]
			runs = sorted(runs, key=seed_sort_fn)
			runs = list(filter(self.filter_fn, runs))

			outliers = []
			for run_name in runs:
				try:
					run = load_svinet_runs(
						path=pjoin(path, run_name),
						true_nodes=list(self.n2id),
					)
				except (ValueError, FileNotFoundError, IndexError):
					# TODO: make logger and record these there
					if self.verbose:
						print('warning, problem with ', run_name)
					continue
				if len(run['community_sv']) == self.num_k:
					self._runs[p][run_name] = run
				else:
					outliers.append(len(run['community_sv']))
			if self.verbose:
				msg = '[PROGRESS] experiment loaded: '
				msg += f"'k = {self.num_k}, {p}',  "
				msg += f"total # runs:  {len(self._runs[p])}"
				print(msg)
			if len(outliers):
				outliers = collections.Counter(outliers).most_common()
				outliers = ',  '.join([f"{n} x k = {k}" for k, n in outliers])
				if self.verbose:
					msg = f"[PROGRESS] some runs"
					msg += f" were excluded:\t{outliers}"
					print(msg)
			# self._filter_runs(**kwargs)
		return self

	def _filter_runs(
			self,
			key: str = 'nll',
			window: int = 5,
			q: float = 0.0, ):
		q %= 1
		if not 0 < q < 1:
			return
		valid_vals = collections.defaultdict(list)
		for p, runs in self._runs.items():
			for run_name, run in runs.items():
				valid_vals[p].append(np.mean(run['valid'][key][-window:]))
		quantiles = {
			p: np.quantile(v, q)
			for p, v in valid_vals.items()
		}
		runs_filtered = collections.defaultdict(dict)
		for p, runs in self._runs.items():
			_accepted = {}
			for run_name, run in runs.items():
				if np.mean(run['valid'][key][-window:]) > quantiles[p]:
					_accepted[run_name] = run
			runs_filtered[p] = _accepted
		self._runs = runs_filtered
		if self.verbose:
			print(f"[PROGRESS] using top {int((1 - q) * 100)} % of runs")
		return

	def _setup_cmaps(self):
		cmap_all, colors = get_cluster_cmap(
			color_ids=range(self.num_k),
			return_colors=True,
		)
		cmaps = {i + 1: get_cluster_cmap(i) for i in range(self.num_k)}
		self.cmaps = {**cmaps, -1: cmap_all}
		self.colors = {i + 1: c for i, c in enumerate(colors)}
		# useful markers:
		# [',', '.', 'o', 'v', '^', '<', '>',
		# '8', 's', 'p', '*', 'h', 'H', 'D',
		# 'd', 'P', 'X']
		mrk = ['o', 'X', 'P', 's', 'D', 'p', 'v', '^', '<', '>']
		self.markers = mrk[:self.num_k]

	def _perms(self, data=None, dist_metric: str = None):
		# TODO: this function only works when there is a single p in _runs
		#  If there are multiple, then _cat_data needs to be modified later
		if data is None:
			data = self._cat_data(
				raw=True, which=self.match_using,
			)
		if dist_metric is None:
			dist_metric = self.match_metric
		# noinspection PyTypeChecker
		dists = sp_dist.cdist(
			XA=self.centroids,
			XB=data,
			metric=dist_metric,
		)
		dists = dists.reshape((self.num_k, -1, self.num_k))
		perms = collections.defaultdict(dict)
		for p, run_dict in self._runs.items():
			for i, run_name in enumerate(run_dict):
				mapping = find_best_match(
					scores=dists[:, i, :],
					reverse=False,
					global_order=True,
				)
				perms[p][run_name] = mapping
		if self.verbose:
			print('[PROGRESS] find best mappings: done')
		return perms

	def _cat_data(self, raw: bool = True, which: str = 'pi'):
		x = []
		if raw:
			for runs_dict in self._runs.values():
				for run in runs_dict.values():
					x.append(run[which])
		else:
			for run in self.get_result_dict().values():
				x.append(np.expand_dims(run[which], axis=0))
		return np.concatenate(x)

	def _align(self):
		for p, _d in self._runs.items():
			for run_name, run in _d.items():
				mapping = self.perms[p][run_name]
				mapping_inv = {j: i for i, j in mapping.items()}

				community = {
					mapping_inv[idx] + 1: [self.n2id[e] for e in comm_list]
					for idx, comm_list in run['community_sv'].items()
				}
				community = dict(sorted(community.items()))
				gamma = {
					self.n2id[i]: g[list(mapping.values())]
					for i, g in run['gamma_sv'].items()
				}
				membership = {
					self.n2id[i]: mem[list(mapping.values())]
					for i, mem in run['membership_sv'].items()
				}
				partition = {
					self.n2id[i]: mapping_inv[idx] + 1
					for i, idx in run['partition_sv'].items()
				}
				gam = run['gam'][list(mapping.values())]
				lam = run['lam'][list(mapping.values())]
				pi = run['pi'][list(mapping.values())]

				_partition = {
					i: comm_idx for i, comm_idx
					in enumerate(np.argmax(pi, axis=0))
				}
				_pi = np.concatenate([
					np.reshape(item, (1, -1))
					for item in membership.values()
				]).T
				_iter = zip(partition.values(), _partition.values())
				if self.verbose:
					# TODO: Make a logger for this stuff
					#  type 1 cause: all zeros in pi
					#  e.g. for k = 4: [0, 0, 0, 0]
					if np.sum([i - 1 != j for i, j in _iter]) > 3:
						print(f'warning type 1 at {run_name}')
					if np.abs(_pi - pi).sum() > 1e-10:
						print('warning type 2')

				run_aligned = {
					'valid': run['valid'],
					'community': community,
					'membership': membership,
					'partition': partition,
					'gamma': gamma,
					'num_missing': run.get('num_missing'),
					'num_overlap': run.get('num_overlap'),
					'gam': gam,
					'lam': lam,
					'pi': pi,
				}
				key = f"{run_name}-{p}"
				self.results[self.perc].update({key: run_aligned})
		if self.verbose:
			print('[PROGRESS] results aligned: done')
		return

	def make_clusters(self, pi: np.ndarray = None):
		# x3d, _ = masks2arr(
		# 	masks=self.project_vec(pi),
		# 	dims_flat=self.mice.al.dims_flat,
		# )
		x3d = self.project_vec(pi)  # TODO
		c_cluster = {
			idx: c for idx, c in self.colors.items()
			if idx in np.unique(x3d)
		}
		m_struct = self.mice.parcel.results['major_masks']
		m_struct = self.mice.al.get_masks(m_struct)
		self.clusters = Clusters(
			partition=x3d,
			m_struct=m_struct,
			c_cluster=c_cluster,
			c_struct=self.mice.parcel.results['major_colors'],
			display='regions',
			threshold=0.0,
		)
		return

	def _trim(self, data, mask):
		if isinstance(data, dict):
			return {
				idx: self.mice.parcel.trim_boundaries_2d(
					x=x, mask=mask.astype(bool),
				) for idx, x in data.items()
			}
		elif isinstance(data, np.ndarray):
			return self.mice.parcel.trim_boundaries_2d(
				x=data, mask=mask.astype(bool),
			)

	def scores(self, decimals: int = 2):
		table = PrettyTable(field_names=['type', 'over', 'score'])
		table.align['type'] = 'l'

		pi_sym_scores = self.symmetry_score(decimals=decimals)
		for idx, s in pi_sym_scores.items():
			row = [
				'pi symmetry',
				'avg' if idx == -1 else f"oc{idx}",
				s,
			]
			table.add_row(row)
		table.add_row(['---'] * len(table.field_names))

		row = [
			'entropy score',
			'nodes',
			self.entropy_score(decimals=decimals),
		]
		table.add_row(row)
		table.add_row(['---'] * len(table.field_names))

		node_scores = self.node_similarity_score(decimals=decimals)
		for metric, s in node_scores.items():
			row = [
				metric,
				'nodes',
				s,
			]
			table.add_row(row)
		print(table)
		return

	@staticmethod
	def node_similarity_score(df=None, decimals: int = 2):
		# df = self.perf_df
		# TODO: later fix this
		scores = {
			metric: df.loc[
				(df.oc_idx == 'all') &
				(df.metric == metric),
				'score'].mean()
			for metric in df.metric.unique()
		}
		return {
			metric: np.round(s, decimals)
			for metric, s in scores.items()
		}

	def entropy_score(self, decimals: int = 2):
		nans = np.isnan(self.h)
		s = 1 - sp_dist.cosine(
			u=self.h[~nans],
			v=self.h_all.mean(0)[~nans],
		)
		return np.round(s, decimals)

	def get_save_path(self, mode: str = 'figs'):
		raise NotImplementedError

	def _add_sup(self, fig, text: str = None, fontsize: int = 20):
		t = text if text else f"{self.base_name}\n\n"
		sup = fig.suptitle(t, fontsize=fontsize)
		return fig, sup

	def plot_performance(self, df, show: bool = True):
		fig, ax = create_figure(1, 1, (13, 6), 'whitegrid')
		sns.boxenplot(
			data=df,  # self.perf_df,
			x='oc_idx',
			y='score',
			hue='metric',
			width=0.7,
			palette='muted',
			ax=ax,
		)
		ax.set_ylim(0, 1)
		plt.legend(bbox_to_anchor=(1.0, 1.01), loc='upper left')

		msg = 'overall:   avg = {:.2f},  median = {:.2f}'
		msg += ' ' * 15
		msg += '{:s}:   avg = {:.2f},  median = {:.2f}'
		cond = df.metric == self.match_metric
		msg = msg.format(
			df.score.mean(),
			df.score.median(),
			self.match_metric,
			df.loc[cond, 'score'].mean(),
			df.loc[cond, 'score'].median(),
		)
		ax.set_title(msg, fontsize=15, y=1.05)
		ax.tick_params('both', labelsize=15)

		file = 'performance.pdf'
		file = pjoin(self.get_save_path(), file)
		fig.savefig(file, bbox_inches='tight')
		if show:
			plt.show()
		else:
			plt.close()
		return fig, ax

	def plot_memberships(
			self,
			kind: str = 'bar',
			stat: str = 'percent',
			show: bool = True,
			save: bool = False,
			**kwargs, ):
		defaults = {
			'nrows': 2,
			'cmap': 'turbo',
			'saturation': 0.7,
			'figsize': (13.5, 11),
			'sup_fontsize': 20,
			'sup_text': None,
			'interpolation': 'none',
			'ci': 95,
			'estimator': np.median,
			'errwidth': 1.5,
			'capsize': 0,
			'size': 5,
			'alpha': 0.10,
			'linewidth': 0.0,
			'xtick_decimals': 2,
			'xtick_rotation': -90,
			'xtick_labelsize': 8,
			'ytick_labelsize': 12,
			'xlabel_fontsize': 14,
			'ylabel_fontsize': 14,
			'title_fontsize': 17,
			'title_y': 1.03,
			'ylim_base': 10,
			'ylim': None,
			'inset_x0': 0.275,
			'inset_y0': 0.40 if kind in ['box', 'bar'] else 0.45,
			'inset_w': 0.5,
			'inset_h': 0.5,
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v

		num_k = len(self.pi_cut_df.oc_idx.unique())
		ncols = int(np.ceil(num_k / kwargs['nrows']))
		fig, axes = create_figure(
			nrows=kwargs['nrows'],
			ncols=ncols,
			figsize=kwargs['figsize'],
			style='white',
			sharex='all',
			sharey='all',
			reshape=True,
		)
		axes_inset = np.empty_like(axes)
		bounds_inset = [
			kwargs['inset_x0'],
			kwargs['inset_y0'],
			kwargs['inset_w'],
			kwargs['inset_h'],
		]

		bins = []
		for intvl in self.pi_cut_df.interval.unique():
			bins.append(intvl.left)
			bins.append(intvl.right)
		bins = np.unique(bins)
		xticklabels = [
			np.round(x, decimals=kwargs['xtick_decimals'])
			for x in bins
		]
		xticks = np.arange(-0.5, len(xticklabels) - 0.5)
		pal = sns.color_palette(kwargs['cmap'], n_colors=len(bins))

		for ctr, oc_idx in enumerate(sorted(self.pi_cut_df.oc_idx.unique())):
			i, j = ctr // ncols, ctr % ncols
			selected_df = self.pi_cut_df.loc[self.pi_cut_df.oc_idx == oc_idx]

			if kind == 'bar':
				sns.barplot(
					x='interval',
					y=stat,
					data=selected_df,
					palette=pal,
					ci=kwargs['ci'],
					saturation=kwargs['saturation'],
					estimator=kwargs['estimator'],
					errwidth=kwargs['errwidth'],
					capsize=kwargs['capsize'],
					ax=axes[i, j],
				)
			elif kind == 'box':
				sns.boxenplot(
					x='interval',
					y=stat,
					data=selected_df,
					palette=pal,
					saturation=kwargs['saturation'],
					ax=axes[i, j],
				)
			elif kind in ['box-strip', 'box-swarm']:
				if kind.split('-')[-1] == 'strip':
					sns.stripplot(
						x='interval',
						y=stat,
						data=selected_df,
						palette=pal,
						size=kwargs['size'],
						alpha=kwargs['alpha'],
						linewidth=kwargs['linewidth'],
						ax=axes[i, j],
					)
				else:
					sns.swarmplot(
						x='interval',
						y=stat,
						data=selected_df,
						palette=pal,
						size=kwargs['size'],
						alpha=kwargs['alpha'],
						linewidth=kwargs['linewidth'],
						ax=axes[i, j],
					)
				props = {
					'boxprops': {'facecolor': 'none', 'edgecolor': 'k'},
					'medianprops': {'color': 'k'},
					'whiskerprops': {'color': 'k'},
					'capprops': {'color': 'k'}
				}
				sns.boxplot(
					x='interval',
					y=stat,
					data=selected_df,
					whis=np.inf,
					saturation=kwargs['saturation'],
					ax=axes[i, j],
					**props,
				)
			else:
				msg = f"invalid kind: {kind}"
				raise RuntimeError(msg)

			# add insets
			x = np.ma.masked_where(self.contour2d, self.pi2d[oc_idx])
			_ax = add_ax_inset(
				data=x,
				ax=axes[i, j],
				bounds=bounds_inset,
				cmap=kwargs['cmap'],
				interpolation=kwargs['interpolation'],
			)
			axes_inset[i, j] = _ax

			# ticks
			axes[i, j].set(
				xticks=xticks,
				xticklabels=xticklabels,
				ylim=np.maximum(0.0, axes[i, j].get_ylim()),
			)
			axes[i, j].tick_params(
				axis='y',
				labelsize=kwargs['ytick_labelsize'],
			)
			axes[i, j].tick_params(
				axis='x',
				rotation=kwargs['xtick_rotation'],
				labelsize=kwargs['xtick_labelsize'],
			)
			axes[i, j].grid()

			# labels
			axes[i, j].set_title(
				label=f"OC{oc_idx}",
				fontsize=kwargs['title_fontsize'],
				y=kwargs['title_y'],
			)
			if i == kwargs['nrows'] - 1:
				axes[i, j].set_xlabel(
					xlabel='Membership',
					fontsize=kwargs['xlabel_fontsize'],
				)
			else:
				axes[i, j].set_xlabel('')
			if j == 0:
				axes[i, j].set_ylabel(
					ylabel=stat.capitalize(),
					fontsize=kwargs['ylabel_fontsize'],
				)
			else:
				axes[i, j].set_ylabel('')

		if kwargs['ylim'] is not None:
			axes[-1, -1].set_ylim(kwargs['ylim'])
		else:
			if kind == 'bar':
				upper = 40
			else:
				upper = self.pi_cut_df.percent.max()
			upper = np.ceil(upper / kwargs['ylim_base'])
			upper *= kwargs['ylim_base']
			axes[-1, -1].set_ylim((0, upper))

		remove_ticks(axes_inset)
		trim_axs(axes, num_k)

		fig, sup = self._add_sup(fig, kwargs['sup_text'], kwargs['sup_fontsize'])
		if save:
			file = [
				f"memberships-dist-{kind}-{stat}",
				f"{'cortex' if True else 'brain'}",
			]
			file = '_'.join(file) + '.pdf'
			file = pjoin(self.get_save_path(), file)
			fig.savefig(
				fname=file,
				bbox_inches='tight',
				bbox_extra_artists=[sup],
			)
		if show:
			plt.show()
		else:
			plt.close()
		output = {
			'fig': fig,
			'sup': sup,
			'axes': axes,
			'axes_inset': axes_inset,
		}
		return output

	def plot_communities_2d(
			self,
			disjoint: bool = True,
			show: bool = True, ):

		nrows = 1
		ncols = int(np.ceil(self.num_k / nrows))
		if nrows * ncols < self.num_k + 1:
			ncols += 1
		width_ratios = [1.] * ncols
		if nrows == 1:
			width_ratios[-1] = 1.2
		fig, axes = create_figure(
			nrows=nrows,
			ncols=ncols,
			figsize=(13, 3),
			gridspec_kw={'width_ratios': width_ratios},
		)
		x2d_dict = self.disjoint2d if disjoint else self.overlap2d
		included = [
			int(e) - 1
			for e in np.unique(x2d_dict[-1])
			if e in self.colors
		]
		for ax, (idx, x2d) in zip(axes, x2d_dict.items()):
			data2plt = np.ma.masked_where(self.contour2d, x2d)
			cmap = get_cluster_cmap(included) if idx == -1 \
				else self.cmaps[idx]
			ax.imshow(data2plt, cmap=cmap)
			ax.set_title(
				label='all' if idx == -1 else
				f"{'C' if disjoint else 'OC'}{idx}",
				fontsize=15 if idx == -1 else 13,
			)
		remove_ticks(axes)
		file = [
			'communities',
			f"{'disjoint' if disjoint else 'overlap'}",
			'2d',
		]
		file = '_'.join(file) + '.pdf'
		file = pjoin(self.get_save_path(), file)
		fig.savefig(file, bbox_inches='tight')
		if show:
			plt.show()
		else:
			plt.close()
		return fig, axes

	def plot_membership_heatmap(self, show: bool = True, **kwargs):
		defaults = {
			'cmap': 'Spectral_r',
			'figsize': (13.5, 4.7),
			'square': False,
			'fmt': '.0f',
			'linewidths': 1,
			'annot_fontsize': 17,
			'cbar_pad': 0.025,
			'xtick_labelsize': 13,
			'ytick_labelsize': 10,
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v
		fig, ax = create_figure(1, 1, kwargs['figsize'], style='white')

		df_table = self.pi_cut_df.groupby(['oc_idx', 'interval']).mean()
		df_table = df_table.unstack()

		sns.heatmap(
			data=df_table.loc[:, 'percent'],
			cmap=kwargs['cmap'],
			annot=True,
			square=kwargs['square'],
			fmt=kwargs['fmt'],
			linewidths=kwargs['linewidths'],
			annot_kws={'fontsize': kwargs['annot_fontsize']},
			cbar_kws={'pad': kwargs['cbar_pad']},
			ax=ax,
		)
		yticklabels = [f"OC{i + 1}" for i in range(len(df_table))]
		ax.set_yticklabels(yticklabels)
		ax.tick_params('x', labelsize=kwargs['xtick_labelsize'])
		ax.tick_params('y', labelsize=kwargs['ytick_labelsize'])
		ax.set_ylabel('')

		file = 'memberships-heatmap.pdf'
		file = pjoin(self.get_save_path(), file)
		fig.savefig(file, bbox_inches='tight')
		if show:
			plt.show()
		else:
			plt.close()
		return fig, ax

	def plot_swarm(
			self,
			x: str = 'entropy',
			show: bool = True,
			**kwargs, ):
		defaults = {
			'sensory_idxs': [2, 3],
			'kind': 'swarm',
			'aspect': 1.2,
			's': 12,
			'title_fontsize': 15,
			'x_fontsize': 15,
			'xtick_labelsize': 13,
			'ytick_labelsize': 14,
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v

		_, df_nodes = self.get_link_node_df()
		if x == 'entropy':
			median = np.median(self.h[~np.isnan(self.h)])
		else:
			_df = self.get_centrality_df()
			_df = _df.loc[_df.measure == 'degree_w']
			median = _df.centrality.median()

		sns.set_theme(style='ticks')
		fg = sns.catplot(
			x=x,
			y='tag',
			data=df_nodes,
			hue="c_idx",
			col='c_type',
			order=sorted(df_nodes.tag.unique()),
			palette=self.colors,
			kind=kwargs['kind'],
			aspect=kwargs['aspect'],
			s=kwargs['s'],
		)
		for ax in fg.axes.ravel():
			ax.axvline(
				x=median,
				ls='--',
				lw=2,
				alpha=1.0,
				c=sns.color_palette('deep')[7],
				label=f"{x} median: {median:.2f}",
				zorder=0,
			)
			ax.tick_params('x', labelsize=kwargs['xtick_labelsize'])
			ax.tick_params('y', labelsize=kwargs['ytick_labelsize'])
			ax.title.set_fontsize(kwargs['title_fontsize'])
			ax.set_xlabel(x, fontsize=kwargs['x_fontsize'])
			ax.set_ylabel('')
			ax.grid(axis='y')

		file = f"swarm_{x}.pdf"
		file = pjoin(self.get_save_path(), file)
		fg.savefig(file, bbox_inches='tight')
		if show:
			plt.show()
		else:
			plt.close()
		return fg, df_nodes

	def plot_4block_percentages(self, show: bool = True, **kwargs):
		defaults = {
			'figsize': (13, 4.2),
			'tick_labelsize': 13,
			'cmap_rot': 0.4,
			'cmap_dark': 0.0,
			'cmap_light': 0.95,
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v

		_, df = self.get_link_node_df()
		percentage1 = collections.defaultdict(dict)
		percentage2 = collections.defaultdict(dict)
		for tag in df.tag.unique():
			values1 = {}
			values2 = {}
			for idx in range(1, self.num_k + 1):
				_df = df.loc[
					(df.tag == tag) &
					(df.c_idx == idx)
					]
				k = f"C{idx}"
				v1 = np.divide(len(_df), len(df.loc[df.tag == tag]))
				v2 = np.divide(len(_df), len(df.loc[df.c_idx == idx]))
				values1[k] = 100 * v1
				values2[k] = 100 * v2
			percentage1[tag] = values1
			percentage2[tag] = values2
		percentage1 = pd.DataFrame(percentage1)
		percentage2 = pd.DataFrame(percentage2)

		fig, axes = create_figure(
			nrows=1, ncols=2,
			figsize=kwargs['figsize'],
			style='white',
			sharex='all',
			sharey='all',
		)
		for i in range(2):
			title = 'row' if i == 0 else 'col'
			x = percentage1 if i == 0 else percentage2
			cmap = sns.cubehelix_palette(
				start=i,
				rot=kwargs['cmap_rot'],
				dark=kwargs['cmap_dark'],
				light=kwargs['cmap_light'],
				reverse=True,
				as_cmap=True,
			)
			sns.heatmap(
				data=x.T.sort_index(),
				annot=True,
				square=True,
				lw=0.1,
				fmt='.0f',
				cmap=cmap,
				ax=axes[i],
			)
			axes[i].set_title(title)
			axes[i].invert_yaxis()
			axes[i].tick_params(
				axis='both',
				labelsize=kwargs['tick_labelsize'],
			)
		file = '4block_percentages.pdf'
		file = pjoin(self.get_save_path(), file)
		fig.savefig(file, bbox_inches='tight')
		if show:
			plt.show()
		else:
			plt.close()
		return fig, axes

	def plot_snr(
			self,
			arrs: List[np.ndarray] = None,
			labels: List[str] = None,
			sort: bool = False,
			sort_idx: int = 0,
			show: bool = True,
			**kwargs, ):
		defaults = {
			'figsize_coef': 1.9,
			'aspect_coef': 0.12,
			'xticks_num': 10,
			'xlabel_fontsize': 14,
			'title_fontsize': 14,
			'title_y': 1.02,
			'cmap_rot': 0.7,
			'cmap_dark': 0.0,
			'cmap_light': 0.95,
			'cbar_pad': 0.01,
			'tight_layout': False,
			'constrained_layout': True,
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v

		if arrs is None:
			pi_mu = np.mean(self.pi_all, axis=0)
			pi_var = np.var(self.pi_all, axis=0)
			pi_mu = np.round(pi_mu, 2)
			pi_var = np.round(pi_var, 2)
			snr = pi_mu ** 2 / np.maximum(pi_var, np.finfo(float).eps)
			snr[np.where(pi_var == 0)] = 0
			arrs = (
				pi_mu,
				pi_var,
				snr,
			)
		labels = labels if labels else (
			r'$\mu$',
			r'$\sigma^2$',
			r'SNR = $\frac{\mu^2}{\sigma^2}$',
		)
		data = dict(zip(labels, arrs))
		fig, axes = create_figure(
			nrows=len(data),
			ncols=1,
			figsize=(13, kwargs['figsize_coef'] * len(data)),
			style='white',
			sharex='col',
			tight_layout=kwargs['tight_layout'],
			constrained_layout=kwargs['constrained_layout'],
		)
		for i, (ax, (lbl, x)) in enumerate(zip(axes, data.items())):
			cmap = sns.cubehelix_palette(
				start=i,
				rot=kwargs['cmap_rot'],
				dark=kwargs['cmap_dark'],
				light=kwargs['cmap_light'],
				reverse=True,
				as_cmap=True,
			)
			if sort:
				x = [
					np.reshape([
						e for _, e in
						sorted(zip(arrs[sort_idx][j], x[j]))
					], (1, -1))
					for j in range(self.num_k)
				]
				x = np.concatenate(x)
			im = ax.imshow(x, vmin=0, cmap=cmap)
			plt.colorbar(im, ax=ax, pad=kwargs['cbar_pad'])
			ax.set_title(
				label=lbl,
				fontsize=kwargs['title_fontsize'],
				y=kwargs['title_y'],
			)
			ax.set(
				xticks=range(0, x.shape[1], kwargs['xticks_num']),
				yticks=range(len(x)),
				yticklabels=[f"oc_{i}" for i in range(1, len(x) + 1)],
				aspect=kwargs['aspect_coef'] / ax.get_data_ratio(),
			)
			if sort:
				ax.set_xlabel('')
			else:
				ax.set_xlabel(
					xlabel='nodes',
					fontsize=kwargs['xlabel_fontsize'],
				)
		file = 'snr_sorted.pdf' if sort else 'snr.pdf'
		file = pjoin(self.get_save_path(), file)
		fig.savefig(file, bbox_inches='tight')
		if show:
			plt.show()
		else:
			plt.close()
		return fig, axes

	def plot_js_vs_weight(self, show: bool = True, **kwargs):
		defaults = {
			'nrows': 2,
			'figsize': (13.5, 10),
			's': 80,
			'alpha': 0.8,
			'title_fontsize': 13,
			'title_y': 1.02,
			'palette': {'between': 'r', 'within': 'k'},
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v

		fig, axes = create_figure(
			nrows=kwargs['nrows'],
			ncols=int(np.ceil(self.num_k / kwargs['nrows'])),
			figsize=kwargs['figsize'],
			style='ticks',
			sharex='all',
			sharey='all',
		)
		df, _ = self.get_link_node_df()
		for ax, idx in zip(axes.ravel(), range(1, self.num_k + 1)):
			sns.scatterplot(
				data=df.loc[df.c_i == idx],
				y='weight',
				x='js_sim',
				hue='link',
				s=kwargs['s'],
				edgecolor=self.colors[idx],
				palette=kwargs['palette'],
				alpha=kwargs['alpha'],
				ax=ax,
			)
			ax.set_title(
				label=f"C{idx}",
				fontsize=kwargs['title_fontsize'],
				y=kwargs['title_y'],
			)
			ax.grid()
		trim_axs(axes, self.num_k)

		file = 'js_vs_weight.pdf'
		file = pjoin(self.get_save_path(), file)
		fig.savefig(file, bbox_inches='tight')
		if show:
			plt.show()
		else:
			plt.close()
		return fig, axes

	def get_centrality_df(self, measures: Union[str, List[str]] = None):
		m = measures if measures else [
			'degree',
			'degree_w',
			'betweenness',
			'betweenness_w',
			'eigenvector',
			'eigenvector_w',
			'closeness',
		]
		m = m if isinstance(m, list) else [m]
		p = self.percentiles[0] if self.perc in [0, -1] else self.perc  # TODO: fix this
		cent_df = self.centralities[p]
		cent_df = cent_df.loc[cent_df.measure.isin(m)]
		cent_df = pd.concat([
			pd.DataFrame({
				'name': [self.perc] * len(self.h),
				'measure': ['entropy'] * len(self.h),
				'node': range(len(self.h)),
				'centrality': self.h, }),
			cent_df,
		])
		cent_df = [cent_df, pd.DataFrame(columns=['oc_idx'])]
		cent_df = pd.concat(cent_df, axis=1)
		partition = {
			n: i + 1 for n, i in
			enumerate(np.argmax(self.pi, axis=0))
		}
		for n, idx in partition.items():
			cent_df.loc[cent_df.node == n, 'oc_idx'] = idx
		return cent_df

	def get_link_node_df(self):
		js_dist = sp_dist.pdist(self.pi.T, metric='jensenshannon')
		js_dist = sp_dist.squareform(js_dist)
		js_sim = 1 - js_dist
		community = {
			idx: [self.id2n[region] for region in x]
			for idx, x in self.get_disjoint_community(to_dict=True).items()
		}
		dict_list = []
		# TODO: Fix this as well
		p = self.percentiles[0] if self.perc in [0, -1] else self.perc
		for i, j, data in self.graphs[p].edges(data=True):
			c_i = next(idx for idx, x in community.items() if i in x)
			c_j = next(idx for idx, x in community.items() if j in x)
			link = 'within' if c_i == c_j else 'between'
			data_dict = {
				'i': [i],
				'j': [j],
				'c_i': [c_i],
				'c_j': [c_j],
				'link': [link],
				'weight': [data['weight']],
				'js_dist': [js_dist[i, j]],
				'js_sim': [js_sim[i, j]],
			}
			dict_list.append(data_dict)
		df_links = pd.DataFrame(merge_dicts(dict_list))

		df = self.get_centrality_df()
		degree = df.loc[df.measure == 'degree_w', 'centrality'].values
		median_degree = np.median(degree)
		median_entropy = np.median(self.h[~np.isnan(self.h)])
		dict_list = []
		# TODO: Fix this as well as well
		p = self.percentiles[0] if self.perc in [0, -1] else self.perc
		for n in self.graphs[p].nodes():
			c = next(idx for idx, x in community.items() if n in x)
			c_type = 'sensory' if c in [2, 3] else 'association'
			deg = degree[n]
			ent = self.h[n]
			if np.isnan(ent):
				continue
			tag = ' / '.join([
				f"{'L-deg' if deg < median_degree else 'H-deg'}",
				f"{'L-ent' if ent < median_entropy else 'H-ent'}",
			])
			data_dict = {
				'node': [n],
				'c_idx': [c],
				'c_type': [c_type],
				'degree': [deg],
				'entropy': [ent],
				'tag': [tag],
			}
			dict_list.append(data_dict)
		df_nodes = pd.DataFrame(merge_dicts(dict_list))
		return df_links, df_nodes

	def make_centrality_plots(self):
		save_dir = self.get_save_path('centrality_plots')
		from scipy.linalg import LinAlgError

		# pair
		selected_measures = [
			'degree_w',
			'closeness',
			'betweenness_w',
			'eigenvector_w',
		]
		df = self.get_centrality_df(selected_measures)
		x = df[['node', 'oc_idx']].drop_duplicates().oc_idx
		_df = df.groupby(['node', 'measure']).mean().unstack()
		_df = _df.centrality
		_df.insert(0, 'oc_idx', x)
		if self.num_k < 11:
			sns.set_theme(style='whitegrid', palette='deep')
			pg = sns.pairplot(
				data=_df,
				hue='oc_idx',
				markers=self.markers[:len(_df.oc_idx.unique())],
				palette=self.colors,
				plot_kws={'alpha': 0.8, 's': 20, 'edgecolor': 'dimgrey'},
			)
			try:
				pg.map_lower(
					func=sns.kdeplot,
					levels=50,
					color=".2",
					fill=True,
					alpha=0.3,
				)
			except LinAlgError:
				pass
			file = pjoin(save_dir, 'pair.pdf')
			pg.savefig(file, bbox_inches='tight')
			plt.close()

		# density
		df = self.get_centrality_df()

		sns.set_theme(style='white', palette='deep')
		all_measures = df.measure.unique().tolist()
		all_measures.remove('entropy')
		for measure in all_measures:
			_df = df.loc[df.measure.isin([measure, 'entropy'])]
			_df = _df.groupby(['node', 'measure']).mean().unstack()
			_df = _df.centrality

			jg = sns.JointGrid(
				data=_df,
				x='entropy',
				y=measure,
				space=0,
			)
			jg.plot_joint(
				sns.kdeplot,
				fill=True,
				clip=((0, 1), (0, 1.0)),
				thresh=0.,
				levels=20,
				gridsize=200,
				cmap="rocket",
			)
			jg.plot_marginals(
				sns.histplot,
				color='#03051A',
				alpha=0.8,
				bins=40,
			)
			_median = np.median(self.h[~np.isnan(self.h)])
			jg.ax_joint.axvline(
				x=_median,
				ls=':',
				lw=0.8,
				alpha=1.0,
				c=sns.color_palette('deep')[-1],
				label=f"entropy median: {_median:.2f}",
			)
			jg.ax_joint.legend(loc='upper left', prop={'size': 8})

			file = pjoin(save_dir, f"{measure}_density.pdf")
			jg.savefig(file)
			plt.close()

			# scatter
			x = df[['node', 'oc_idx']].drop_duplicates().oc_idx
			_df.insert(0, 'oc_idx', x)

			jg = sns.JointGrid(
				data=_df,
				x='entropy',
				y=measure,
				hue='oc_idx',
				palette=self.colors,
				xlim=(0, 1),
				ylim=(0, 1),
				space=0.0,
			)
			jg.plot_joint(
				sns.scatterplot,
				hue='oc_idx',
				s=180,
				alpha=0.9,
				edgecolor='dimgrey',
			)
			jg.plot_marginals(
				sns.kdeplot,
				fill=True,
				alpha=0.5,
				lw=2,
			)
			_median = np.median(self.h[~np.isnan(self.h)])
			jg.ax_joint.axvline(
				x=_median,
				ls='--',
				lw=2,
				alpha=1.0,
				c=sns.color_palette('deep')[7],
				label=f"entropy median: {_median:.2f}",
				zorder=0,
			)
			_median = df.loc[
				df.measure == measure,
				'centrality'
			].median()
			jg.ax_joint.axhline(
				y=_median,
				ls='--',
				lw=2,
				alpha=1.0,
				c=sns.color_palette('deep')[7],
				label=f"{measure} median: {_median:.2f}",
				zorder=0,
			)
			jg.ax_joint.legend(loc='upper left', prop={'size': 9})

			file = pjoin(save_dir, f"{measure}_scatter.pdf")
			jg.savefig(file)
			plt.close()

	def render(self, pi: np.ndarray = None):
		from utils.render import parcel2html
		for disjoint in [False, True]:
			file = '_'.join([
				f"{'disjoint' if disjoint else 'overlap'}",
				f"{'brain'}",
			])
			_ = parcel2html(
				masks=self.project_vec(pi),
				save_dir=self.get_save_path('render'),
				save_name=file,
				cmap=self.cmaps[-1],
				base_name='none',
				alpha=1.0,
				alpha_root=0.1,
			)
		return


# noinspection PyTypeChecker
def _load_sv_validedges(lines: List[str]):
	valid_edges = collections.defaultdict(list)
	for i, line in enumerate(lines):
		x = line.strip('\n').split('\t')
		x = list(filter(None, x))
		if len(x):
			i, j, k = tuple(map(lambda s: int(s), x))
			valid_edges[k].append((i, j))

	def _edgeseed_sort_fn(e):
		n, m = e
		return m + 1e5 * n

	valid_edges = dict(sorted([
		(k, sorted(v, key=_edgeseed_sort_fn))
		for k, v in valid_edges.items()
	]))
	return valid_edges


# noinspection PyTypeChecker
def _load_sv_valid(lines: List[str]):
	_vld = collections.defaultdict(dict)
	for i, line in enumerate(lines):
		x = line.strip('\n').split('\t')
		x = list(filter(None, x))
		if len(x):
			_vld[int(x[0])] = {
				'nll': -float(x[2]),
				'n': int(x[3]),
				'nll_link': -float(x[4]),
				'n_link': int(x[5]),
				'nll_nonlink': -float(x[6]),
				'n_nonlink': int(x[7]),
				'nll_sparse_link': -float(x[8]),
				'nll_sparse_nonlink': -float(x[9]),
				'nll_sparse': -float(x[10]),
			}
	_vld = dict(sorted(_vld.items()))
	valid = {
		'n': np.unique([v['n'] for v in _vld.values()]).item(),
		'n_link': np.unique([v['n_link'] for v in _vld.values()]).item(),
		'n_nonlink': np.unique([v['n_nonlink'] for v in _vld.values()]).item(),
		'nll': [v['nll'] for v in _vld.values()],
		'nll_link': [v['nll_link'] for v in _vld.values()],
		'nll_nonlink': [v['nll_nonlink'] for v in _vld.values()],
		'nll_sparse': [v['nll_sparse'] for v in _vld.values()],
		'nll_sparse_link': [v['nll_sparse_link'] for v in _vld.values()],
		'nll_sparse_nonlink': [v['nll_sparse_nonlink'] for v in _vld.values()],
	}
	return valid


def _load_sv_lambda(lines: List[str]):
	lambda_sv = {}
	for line in lines:
		x = line.strip('\n').split('\t')
		x = list(filter(None, x))
		if len(x):
			lambda_sv[int(x[0])] = tuple(float(s) for s in x[1:])
	return lambda_sv


def _load_sv_community(lines: List[str]):
	community_sv = {}
	for i, line in enumerate(lines):
		x = line.strip('\n').split(' ')
		x = list(filter(None, x))
		community_sv[i] = [int(s) for s in x]
	return community_sv


def _load_sv_membership(lines: List[str], true_nodes: list, eps: float = 0.):
	membership_sv = {}
	for line in lines:
		x = line.strip('\n').split('\t')
		node = int(x[1])
		mem = [float(s) for s in x[2:]]
		membership_sv[node] = np.array(mem)
	_num = len(list(membership_sv.values())[0])
	for node in true_nodes:
		if node not in membership_sv:
			membership_sv[node] = np.ones(_num) * eps
	# noinspection PyTypeChecker
	membership_sv = dict(sorted(membership_sv.items()))
	return membership_sv


def load_svinet_runs(path: str, true_nodes: Union[int, range, list]):
	if isinstance(true_nodes, int):
		true_nodes = range(true_nodes)

	# validation data
	with open(pjoin(path, 'validation.txt'), 'r') as f:
		lines = f.readlines()
	try:
		valid = _load_sv_valid(lines)
	except (IndexError, ValueError):
		valid = {}
	with open(pjoin(path, 'validation-edges.txt'), 'r') as f:
		lines = f.readlines()
	try:
		valid = {
			'edges': _load_sv_validedges(lines),
			**valid,
		}
	except ValueError:
		valid = {}

	# fitted parameters
	with open(pjoin(path, 'communities.txt'), 'r') as f:
		lines = f.readlines()
	community_sv = _load_sv_community(lines)
	with open(pjoin(path, 'groups.txt'), 'r') as f:
		lines = f.readlines()
	membership_sv = _load_sv_membership(lines, true_nodes)
	with open(pjoin(path, 'gamma.txt'), 'r') as f:
		lines = f.readlines()
	gamma_sv = _load_sv_membership(lines, true_nodes)
	with open(pjoin(path, 'lambda.txt'), 'r') as f:
		lines = f.readlines()
	lambda_sv = _load_sv_lambda(lines)

	# gamma, pi, lambda, and partition
	gam = np.concatenate([
		np.reshape(x, (1, -1))
		for x in gamma_sv.values()
	])
	lam = np.concatenate([
		np.reshape(x, (1, -1))
		for x in lambda_sv.values()
	])
	pi = np.concatenate([
		np.reshape(x, (1, -1))
		for x in membership_sv.values()
	])
	partition_sv = {
		i: comm_idx for i, comm_idx
		in enumerate(np.argmax(pi, axis=1))
	}

	included = []
	for v in community_sv.values():
		included.extend(v)
	n_included = len(set(included))

	results = {
		'valid': valid,
		'gamma_sv': gamma_sv,
		'lambda_sv': lambda_sv,
		'community_sv': community_sv,
		'membership_sv': membership_sv,
		'partition_sv': partition_sv,
		'num_missing': len(pi) - n_included,
		'num_overlap': len(included) - n_included,
		'gam': gam.T,
		'lam': lam,
		'pi': pi.T,
	}
	return results


def count_sv_fitted(
		path: str,
		perc: str,
		group: str = 'SLC', ):
	total_fitted = collections.defaultdict(dict)
	for p in pathlib.Path(path).rglob(f"**/{perc}"):
		name = next(e for e in str(p).split('/') if group in e)
		b = next(e for e in name.split('_') if 'bp' in e)[2:]
		key = name[name.find('sub'):]
		total_fitted[b][key] = len(os.listdir(p)) - 1
	total_fitted = {
		b: dict(sorted(d.items())) for
		b, d in sorted(total_fitted.items())
	}
	return total_fitted


def _extract_seed(x):
	x = next(
		e for e in
		str(x).split('-')
		if 'seed' in e
	)
	x = re.sub(r"\D", "", x)
	return int(x)


def _get_filter_fn(k, *args):
	keywords = [f"k{k}", *args]

	def filter_fn(x):
		return all(key in x for key in keywords)

	return filter_fn


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
		choices={'columnar', 'allen'},
	)
	parser.add_argument(
		"--k",
		help='number of communities',
		type=int,
		default=5,
	)
	parser.add_argument(
		"--p",
		help='percentile',
		type=str,
		default='p20-sample',
	)
	parser.add_argument(
		"--key",
		help='key to choose data from Mice, default: loop over all',
		type=str,
		default='all',
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
		choices={'pearson', 'spearman', 'correlation', 'cosine'},
		default='pearson',
	)
	parser.add_argument(
		"--match_metric",
		help='distance metric to use with kmeans',
		type=str,
		choices={'cosine', 'euclidean'},
		default='euclidean',
	)
	parser.add_argument(
		"--match_using",
		help='data to use in matching fits',
		type=str,
		choices={'pi', 'gam'},
		default='gam',
	)
	parser.add_argument(
		"--graph_type",
		help='real or a random baseline',
		type=str,
		choices={'real', 'config', 'config_clust'},
		default='real',
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
	props = {
		'mice': mice,
		'num_k': args.k,
		'perc': args.p,
		'mode': args.mode,
		'task': None,
		'metric': args.metric,
		'match_metric': args.match_metric,
		'match_using': args.match_using,
		'graph_type': args.graph_type,
		'verbose': False,
	}
	# setup variables & pbar
	if args.mode == 'ca2':
		bands = [0.01, 0.5, (0.01, 0.5)]
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
		props['task'] = task
		if args.key == 'all':
			pbar2 = tqdm(
				mice.get_data_containers(args.mode)[0],
				position=-1,
				leave=False,
				dynamic_ncols=True,
			)
			for key in pbar2:
				msg = f"running sv (task-{task}, desc-{desc}, "
				msg += f"b-{str(b).replace(' ', '')})"
				msg += f": {key}"
				pbar1.set_description(msg)
				sv = SVINET(key=key, **props)
				sv.fit_svinet(force=False)
		else:
			_ = SVINET(key=args.key, **props).fit_svinet()

	print('\n')
	print(args)
	print(f"[PROGRESS] analysis.svinet done {now(True)}.\n")


if __name__ == "__main__":
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		main()
