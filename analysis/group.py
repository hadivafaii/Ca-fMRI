from .svinet import *
from .network import collect_degrees_lite

_ATTRS = [
	'n_runs',
	'n_seeds',
	'address',
	'fit_meta',  # dicts
	'run_pi',
	'run_lam',
	'run_deg',
	'run_seed_rank',  # arrs
	'pi_prune',  # from boot
]


class Group(Base):
	def __init__(
			self,
			perc: str = 'p20-sample',
			sv_props: dict = None,
			graph_type: str = 'real',
			align_metric: str = 'cosine',
			ordering: str = 'infer',
			kw_kmeans: dict = None,
			**kwargs,
	):
		super(Group, self).__init__(**kwargs)
		self.perc = perc
		self.graph_type = graph_type
		self.sv_props = sv_props if sv_props else {
			'match_metric': 'euclidean',
			'match_using': 'gam',
		}
		self.align_metric = align_metric
		if ordering == 'infer':
			self.global_order = 'infer'
		elif ordering == 'global':
			self.global_order = True
		elif ordering == 'local':
			self.global_order = False
		else:
			raise ValueError(f"invalid ordering: '{ordering}'")
		self.ordering = ordering

		kmeans_defaults = {
			'n_init': 100,
			'max_iter': 300,
			'tol': 1e-4,
		}
		kw_kmeans = kw_kmeans if kw_kmeans else {}
		for k, v in kmeans_defaults.items():
			if k not in kw_kmeans:
				kw_kmeans[k] = v
		self.kw_kmeans = kw_kmeans

		self.svinets = {}
		self.fit_meta = {}
		self.perms = collections.defaultdict(dict)
		self.pi_prune = None
		self.load_group()

	def _set_attrs(self, results: dict):
		self.n_runs = results.get('n_runs', {})
		self.n_seeds = results.get('n_seeds', {})
		self.address = results.get('address', {})
		self.run_pi = results.get('pi_run')
		self.run_lam = results.get('lam_run')
		self.run_deg = results.get('deg_run')
		self.run_seed_rank = results.get('seed_rank')
		return

	def _add_mode_df(self, df):
		df.insert(0, 'mode', self.mode)

	def fit_group(self, force: bool = False):
		if not force:
			if self.fit_meta:
				return self
			else:
				if self.verbose:
					print('Fit not found . . . initiating fit')

		_ = self.load_svinets()
		if not len(self.svinets):
			if self.verbose:
				print('SVINETs not loaded, probably wrong params . . . ')
			return self

		self.fit_meta['time_start'] = now(True)
		self.perms, _ = self.align_group()
		data = self._apply_perms()
		results_agg = self._aggregate_data(data)
		self.fit_meta['time_end'] = now(True)
		# results = {
		# **results_agg,  # TODO: fix prune, cast vote etc
		# **self.prune(pi_from_gam(results_agg['gam_run'])),
		# **self.get_entropy_degree_grp(results_agg['gam_run']),
		# }
		self._set_attrs(results_agg)
		self._save_results()
		return self

	def load_group(self):
		for name in _ATTRS:
			x = pjoin(self.path(False), f"{name}.npy")
			if name in ['n_runs', 'n_seeds', 'address', 'fit_meta']:
				try:
					x = np.load(x, allow_pickle=True)
					setattr(self, name, x.item())
				except FileNotFoundError:
					return
			else:
				try:
					x = np.load(x, mmap_mode='r')
					setattr(self, name, x)
				except FileNotFoundError:
					continue
		return self

	def apply_global_perms(self, gmap: Dict[int, int]):
		ids = list(gmap.values())
		for attr in ['run_lam', 'run_pi', 'pi_prune', 'pi']:
			try:
				x = getattr(self, attr)
				if x is None:
					continue
				setattr(self, attr, x[..., ids, :])
			except AttributeError:
				continue
		return

	def _save_results(self):
		for item in _ATTRS:
			x = getattr(self, item)
			if x is None:
				continue
			save_obj(
				obj=x,
				file_name=item,
				save_dir=self.path(True),
				verbose=self.verbose,
				mode='npy',
			)
			setattr(self, item, None)
		return

	def _apply_perms(self):
		permuted = collections.defaultdict(dict)
		for key, sv in self.svinets.items():
			_pi, _gam, _lam = [], [], []
			for i, mapping in self.perms[key].items():
				ids = list(mapping.values())
				_pi.append(sv.pi_all[[i]][:, ids])
				_gam.append(sv.gam_all[[i]][:, ids])
				_lam.append(sv.lam_all[[i]][:, ids])
			permuted[key] = {
				'pi': np.concatenate(_pi),
				'gam': np.concatenate(_gam),
				'lam': np.concatenate(_lam),
			}
		return permuted

	def _aggregate_data(self, data: Dict[str, dict], runs_only: bool = True):
		n_seeds = {}
		for key, sv in self.svinets.items():
			n_seeds[key] = len(sv.gam_all)

		combine_keys = self.mice.combine_keys(self.mode)
		n_runs = {}
		for key, (sub, ses, _) in self.mice.looper_agg.items():
			if sub == -1 or ses == -1:
				continue
			n_runs[key] = len(combine_keys[key])

		pi_runs = np_nans((
			len(self.mice.cfg.sub_ids),
			len(self.mice.cfg.ses_ids),
			max(n_runs.values()),
			max(n_seeds.values()),
			self.num_k,
			self.num_n,
		))
		lam_runs = np_nans((*pi_runs.shape[:-1], 2))

		if not runs_only:
			pi_sessions = np_nans((
				len(self.mice.cfg.sub_ids),
				len(self.mice.cfg.ses_ids),
				max(n_seeds.values()),
				self.num_k,
				self.num_n,
			))
			lam_sessions = np_nans((*pi_sessions.shape[:-1], 2))
		else:
			pi_sessions = None
			lam_sessions = None

		degree = self.collect_degrees()
		deg_runs = np_nans((
			len(self.mice.cfg.sub_ids),
			len(self.mice.cfg.ses_ids),
			max(n_runs.values()),
			self.num_n,
		))

		address = {}
		for key, (sub, ses, _) in self.mice.looper_agg.items():
			if sub == -1 or ses == -1:
				continue
			if key in self.svinets and not runs_only:
				ids = range(n_seeds[key])
				pi_sessions[sub - 1, ses - 1][ids] = data[key].pop('pi')
				lam_sessions[sub - 1, ses - 1][ids] = data[key].pop('lam')
			else:
				if self.verbose and not runs_only:
					print(f"Warning! {key} was not fit.  Moving on . . .")

			for i, run_k in enumerate(combine_keys[key]):
				if run_k not in self.svinets:
					if self.verbose:
						print(f"Warning! {run_k} was not fit.  Moving on . . .")
					continue
				ids = (
					self.mice.cfg.sub_ids.index(sub),
					self.mice.cfg.ses_ids.index(ses),
					i,
				)
				seed_ids = range(n_seeds[run_k])
				pi_runs[ids][seed_ids] = data[run_k].pop('pi')
				lam_runs[ids][seed_ids] = data[run_k].pop('lam')
				deg_runs[ids] = degree.pop(run_k)
				address[run_k] = ids

		seed_rank, _ = self._rank_seeds(pi_runs)
		i, j, k = pi_runs.shape[:3]
		looper = itertools.product(
			range(i), range(j), range(k))
		for i, j, k in looper:
			r = seed_rank[i, j, k]
			pi_runs[i, j, k] = pi_runs[i, j, k][r]
			lam_runs[i, j, k] = lam_runs[i, j, k][r]
		output = {
			'n_runs': n_runs,
			'n_seeds': n_seeds,
			'address': address,
			'pi_run': pi_runs,
			'lam_run': lam_runs,
			'deg_run': deg_runs,
			'pi_ses': pi_sessions,
			'lam_ses': lam_sessions,
			'seed_rank': seed_rank,
		}
		return output

	def collect_degrees(self):
		rho = int(re.findall(
			pattern=r"\d+",
			string=self.perc.split('-')[0]
		).pop()) / 100
		return collect_degrees_lite(
			mice=self.mice,
			mode=self.mode,
			rho=rho,
		)

	def align_group(self):
		accepted_runs, good_global = self.find_good_nodes()
		sv_attr = f"{self.sv_props['match_using']}_all"
		data = [
			getattr(sv, sv_attr).reshape(-1, self.num_n)
			for k, sv in self.svinets.items()
			if k in accepted_runs
		]
		data = np.concatenate(data)
		nonan = np.where(np.isnan(data).sum(0) == 0)[0]
		centroids = np.zeros((self.num_k, data.shape[-1]))
		_centroids, _ = fit_kmeans(
			data=data[:, nonan],
			n_clusters=self.num_k,
			match_metric=self.sv_props['match_metric'],
			random_state=self.mice.cfg.random_state,
			kw_kmeans=self.kw_kmeans,
		)
		centroids[:, nonan] = _centroids

		global_mapping = self.align_centroid_to_structs(
			pi_mv=centroids,
			metric='cosine',
			global_order=False,
		)
		centroids = centroids[global_mapping]
		ner = Normalizer()
		perms = {}
		for k, sv in self.svinets.items():
			good = getattr(sv, self.sv_props['match_using'])
			good = np.where(np.isnan(good).sum(0) == 0)[0]
			good = set(good).intersection(good_global)
			good = np.array(sorted(good))
			_x = getattr(sv, sv_attr)[..., good]
			_x = np.where(~np.isnan(_x), _x, 0.0)
			data_clean = {
				i: ner.fit_transform(_x[i])
				for i in range(len(_x))
			}
			perms[k] = find_best_mappings(
				data=data_clean,
				centroids=ner.fit_transform(centroids[:, good]),
				match_metric=self.align_metric,
				global_order=self.global_order,
			)
		return perms, centroids

	def find_good_nodes(
			self,
			data: Union[np.ndarray, dict] = None,
			mide_node_tolerance: float = None, ):
		if not mide_node_tolerance:
			mide_node_tolerance = 0.50 if self.mode == 'ca2' else 0.55
		min_runs_thresh = 0.60 if self.mode == 'ca2' else 0.75
		min_node_thresh = 0.70 if self.mode == 'ca2' else 0.85
		results = {}
		for q in np.linspace(0.8, 1, 41):
			accepted_runs, good_global, tot = self._good_nodes(data, q)
			results[q] = (
				len(accepted_runs) / tot,
				len(good_global) / self.num_n,
			)
		safety = 0
		best_q = None
		while best_q is None:
			best_r = -np.inf
			for q, (r1, r2) in results.items():
				if r1 > min_runs_thresh and r2 > min_node_thresh:
					r = (r1 + r2) * 0.5
					if r > best_r:
						best_r = r
						best_q = q
			if best_q is None:
				if min_node_thresh > mide_node_tolerance:
					min_node_thresh -= 0.01
				else:
					min_runs_thresh -= 0.01
			safety += 1
			if safety > 100:
				break
		accepted_runs, good_global, tot = self._good_nodes(data, best_q)
		self.fit_meta['min_runs_thresh'] = min_runs_thresh
		self.fit_meta['min_node_thresh'] = min_node_thresh
		self.fit_meta['best_node_freq_q'] = best_q
		self.fit_meta['num_total_runs'] = tot
		self.fit_meta['fraction_accepted_runs'] = len(accepted_runs) / tot
		self.fit_meta['fraction_good_nodes'] = len(good_global) / self.num_n
		return accepted_runs, good_global

	def _good_nodes(
			self,
			data: Union[np.ndarray, dict] = None,
			freq_q: float = None, ):
		if data is None:
			data = self.svinets if self.svinets else self.load_svinets()
		if freq_q is None:
			freq_q = 0.90 if self.mode == 'ca2' else 0.95
		node_presence = collections.defaultdict(list)
		tot = 0
		if isinstance(data, np.ndarray):
			i, j, k = data.shape[:3]
			looper = itertools.product(
				range(i), range(j), range(k))
			for i, j, k in looper:
				x = data[i, j, k]
				good = np.where(np.isnan(x).sum(0) == 0)[0]
				if not len(good):
					continue
				tot += 1
				for n in good:
					node_presence[n].append((i, j, k))
		elif isinstance(data, dict):
			for key, sv in data.items():
				if 'run' not in key:
					continue
				good = np.where(np.isnan(
					getattr(sv, self.sv_props['match_using'])
				).sum(0) == 0)[0]
				if not len(good):
					continue
				tot += 1
				for n in good:
					node_presence[n].append(key)

		node_presence = dict(sorted(node_presence.items()))
		freq = {
			n: len(v) / tot for n, v
			in node_presence.items()
		}
		good_global = np.array([
			n for n, f in freq.items()
			if f > freq_q
		])
		accepted_runs = []
		if isinstance(data, np.ndarray):
			i, j, k = data.shape[:3]
			looper = itertools.product(
				range(i), range(j), range(k))
			for i, j, k in looper:
				x = data[i, j, k]
				good = np.where(np.isnan(x).sum(0) == 0)[0]
				if all(e in good for e in good_global):
					accepted_runs.append((i, j, k))
		elif isinstance(data, dict):
			for key, sv in data.items():
				if 'run' not in key:
					continue
				good = np.where(np.isnan(sv.gam).sum(0) == 0)[0]
				if all(e in good for e in good_global):
					accepted_runs.append(key)
		return accepted_runs, good_global, tot

	def load_svinets(self):
		svinet_props = {
			'mice': self.mice,
			'num_k': self.num_k,
			'perc': self.perc,
			'mode': self.mode,
			'task': None,
			'metric': self.metric,
			'graph_type': self.graph_type,
			**self.sv_props,
		}
		if self.task == 'all':
			tasks = ['rest', 'led']
		else:
			tasks = [self.task]
		svinets = {}
		for task in tasks:
			svs = {}
			svinet_props['task'] = task
			proc, *_ = self.mice.get_data_containers(
				mode=self.mode, force=True, task=task)
			for key in proc:
				sv = SVINET(
					key=key,
					**svinet_props,
				).load_results()
				if sv is not None:
					svinets[key] = sv
			svinets = {**svinets, **svs}
		self.svinets = dict(sorted(svinets.items()))
		return svinets

	def svinets_exist(self):
		pat = self.name().split('_') + [self.sv_props['match_metric']]
		sv_path = pjoin(self.mice.cfg.results_dir, 'SVINET')
		sv_path = pathlib.Path(sv_path).rglob('**/*')
		try:
			next(
				s for s in sv_path if
				all(e in str(s) for e in pat)
			)
			return True
		except StopIteration:
			return False

	def name(self):
		mode = self.mode.split('-')[0]
		band = self.mice.kws.get(f"band_{mode}")
		desc = self.mice.kws.get(f'desc_{mode}')
		name = [
			self.graph_type,
			self.metric,
			self.task,
			'-'.join([self.mode, desc]),
			f"bp{band}".replace(' ', '')
			if band else None,
			f"k{self.num_k}",
			self.perc,
		]
		name = [e for e in name if e]
		name = '_'.join(name)
		return name

	def _rank_seeds(
			self,
			x: np.ndarray = None,
			axis_seed: int = 3, ):
		if x is None:
			x = self.run_pi
		assert x.ndim == 6, "must have seeds"
		i, j, k = x.shape[:3]
		n_nodes = self.num_n
		n_seeds = x.shape[axis_seed]
		centroids = bn.nanmean(bn.nanmean(bn.nanmean(
			x, axis_seed), 2), 1)
		seed_rank = np.zeros((i, j, k, n_seeds), dtype=int)

		looper = itertools.product(
			range(i), range(j), range(k))
		missing_runs = []
		for i, j, k in looper:
			data = x[i, j, k]
			nan_seeds = np.isnan(data).sum(-1).mean(-1)
			nan_seeds = np.where(nan_seeds == n_nodes)[0]
			good_seeds = np.delete(range(len(data)), nan_seeds)
			if not len(good_seeds):
				missing_runs.append((i, j, k))
				continue
			data = data[good_seeds]
			data_flat = data.reshape(-1, data.shape[-1])
			nonan = np.isnan(data_flat).sum(0) == 0

			dists = np.zeros(len(good_seeds))
			for seed_i in range(len(good_seeds)):
				# noinspection PyTypeChecker
				d = sp_dist.cdist(
					XA=centroids[i][:, nonan],
					XB=data[seed_i][:, nonan],
					metric=self.align_metric,
				)
				dists[seed_i] = np.diag(d).mean()
			r = np.concatenate([
				np.argsort(dists),
				nan_seeds,
			])
			seed_rank[i, j, k] = r
		return seed_rank, missing_runs

	def get_anim_num_runs(self):
		anim_n_runs = collections.defaultdict(int)
		for key in self.address:
			sub, ses, run = self.mice.looper[key]
			anim_n_runs[self.mice.get_key(sub)] += 1
		return dict(anim_n_runs)

	def ttest(
			self,
			se: np.ndarray,
			pi: np.ndarray = None,
			mu: float = 0.003,
			alpha: float = 0.05,
			multi: str = 'fdr_bh', ):

		pi = pi if pi is not None else self.run_pi
		pi = avg(pi, 2, se.ndim - 2)
		assert pi.shape == se.shape

		tvals = (pi - mu) / se
		pvals = np.zeros(pi.shape)
		reject = np.zeros(pi.shape)

		if se.ndim == 3:  # for pruning
			anim_n_runs = self.get_anim_num_runs()
			for i in range(len(pi)):
				anim_key = self.mice.get_key(i + 1)
				dof = anim_n_runs[anim_key] - 1
				p = 1 - sp_stats.t.cdf(tvals[i], dof)
				if multi:
					nonan = np.isfinite(p)
					corrected = multipletests(
						pvals=p[nonan],
						alpha=alpha,
						method=multi,
						is_sorted=False,
						returnsorted=False,
					)
					reject[i][nonan] = corrected[0]
					pvals[i][nonan] = corrected[1]
				else:
					reject[i] = p < alpha
					pvals[i] = p

		elif se.ndim == 2:  # for declaring members
			dof = self.mice.cfg.n_subjects - 1
			p = 1 - sp_stats.t.cdf(tvals, dof)
			if multi:
				nonan = np.isfinite(p)
				corrected = multipletests(
					pvals=p[nonan],
					alpha=alpha,
					method='fdr_bh',
					is_sorted=False,
					returnsorted=False,
				)
				reject[nonan] = corrected[0]
				pvals[nonan] = corrected[1]
			else:
				reject = p < alpha
				pvals = p
		else:
			raise NotImplementedError
		return reject, pvals

	def align_post_prune(self, pi: np.ndarray, save: bool = True):
		accepted_runs, good_global = self.find_good_nodes(
			data=bn.nanmean(pi, -3))
		data = []
		for i, j, k in accepted_runs:
			x = pi[i, j, k]
			bad_seeds = np.isnan(x).sum(-1).sum(-1)
			bad_seeds = bad_seeds == self.num_k * self.num_n
			data.append(x[~bad_seeds].reshape(-1, self.num_n))
		data = np.concatenate(data)

		centroids = np.zeros((self.num_k, data.shape[-1]))
		_centroids, _ = fit_kmeans(
			data=data[:, good_global],
			n_clusters=self.num_k,
			match_metric=self.sv_props['match_metric'],
			random_state=self.mice.cfg.random_state,
			kw_kmeans=self.kw_kmeans,
		)
		centroids[:, good_global] = _centroids
		global_mapping = self.align_centroid_to_structs(
			pi_mv=centroids,
			metric='cosine',
			global_order=False,
		)
		centroids = centroids[global_mapping]

		perms = {}
		ner = Normalizer()
		ii, jj, kk = pi.shape[:3]
		looper = itertools.product(
			range(ii), range(jj), range(kk))
		for i, j, k in looper:
			x = pi[i, j, k]
			bad_seeds = np.isnan(x).sum(-1).sum(-1)
			bad_seeds = bad_seeds == self.num_k * self.num_n
			if bad_seeds.sum() == len(x):
				continue
			x = x[~bad_seeds]
			good = np.where(np.isnan(x).sum(0).sum(0) == 0)[0]
			good = set(good).intersection(good_global)
			good = np.array(sorted(good))

			_x = x[..., good]
			_x = np.where(~np.isnan(_x), _x, 0.0)
			data2align = {
				i: ner.fit_transform(v)
				for i, v in enumerate(_x)
			}
			perms[(i, j, k)] = find_best_mappings(
				data=data2align,
				centroids=ner.fit_transform(centroids[:, good]),
				match_metric=self.align_metric,
				global_order=self.global_order,
			)
		aligned = np_nans(pi.shape)
		looper = itertools.product(
			range(ii), range(jj), range(kk))
		for i, j, k in looper:
			gmap_dict = perms.get((i, j, k))
			if not gmap_dict:
				continue
			for seed, gmap in gmap_dict.items():
				ids = list(gmap.values())
				aligned[i, j, k, seed] = pi[i, j, k, seed][ids]

		# final alignment
		gmap = find_best_mappings(
			data={0: avg(aligned)},
			centroids=avg(self.run_pi),
			match_metric=self.align_metric,
			global_order=self.global_order,
		)[0]
		aligned = aligned[..., list(gmap.values()), :]

		if save:
			save_obj(
				obj=perms,
				file_name='perms_prune',
				save_dir=self.path(),
				verbose=self.verbose,
				mode='npy',
			)
			save_obj(
				obj=aligned,
				file_name='pi_prune',
				save_dir=self.path(),
				verbose=self.verbose,
				mode='npy',
			)
		return aligned, centroids

	def cast_votes(
			self,
			se_anim: np.ndarray,
			pi: np.ndarray = None,
			thres: float = 0.003,
			ci: float = 0.95, ):
		pi = pi if pi is not None else self.run_pi
		i, j, k, seeds, kk, nn = pi.shape
		pi_anim = bn.nanmean(bn.nanmean(bn.nanmean(pi, 3), 2), 1)

		anim_n_runs = self.get_anim_num_runs()
		votes = np.zeros((nn, i, kk))
		rej, acc = 0, 0
		for anim in range(i):
			dof = anim_n_runs.get(self.mice.get_key(anim + 1))
			if dof:
				dof -= 1
			else:
				dof = j * k - 1
			t_val = sp_stats.t.ppf(ci, dof)
			looper = itertools.product(
				range(nn), range(kk))
			for node, oc in looper:
				mu = pi_anim[anim, oc, node]
				se = se_anim[anim, oc, node]
				lower = mu - t_val * se
				if np.isnan(lower):
					votes[node, anim, oc] = 1
				elif lower <= thres or np.isnan(lower):
					votes[node, anim, oc] = 1
					rej += 1
				else:
					acc += 1
		votes_portion = bn.nanmean(votes, 1)
		output = {
			'votes': votes.astype(bool),
			'votes_portion': votes_portion,
			'votes_majority': votes_portion > 0.5,
			'num_accepted': acc,
			'num_zeroed': rej,
		}
		return output

	def avg(
			self,
			x: np.ndarray = None,
			apply: bool = True,
			**kwargs, ):
		defaults = dict(
			ndim_end=2,
			ndim_start=0,
			fn=bn.nanmean,
		)
		kwargs = setup_kwargs(defaults, kwargs)
		x = x if x is not None else self.run_pi
		x_avg = avg(x, **kwargs)
		if apply:
			self.pi = x_avg
		return x_avg


def node_sim_matrix(pi, beta):
	k, n_nodes = pi.shape[-2:]
	p_indiv, b_indiv = tuple(map(
		lambda x: bn.nanmean(bn.nanmean(bn.nanmean(
			x, 3), 2), 1), [pi, beta]
	))

	y = np.zeros((len(p_indiv), n_nodes, n_nodes))
	for i in range(len(p_indiv)):
		diag = np.zeros((k, k))
		np.fill_diagonal(diag, b_indiv[i])
		y[i] = p_indiv[i].T @ diag @ p_indiv[i]

	p_avg, b_avg = tuple(map(
		lambda x: bn.nanmean(x, 0), [p_indiv, b_indiv]
	))

	diag = np.zeros((k, k))
	np.fill_diagonal(diag, b_avg)
	y_avg = p_avg.T @ diag @ p_avg

	return y, y_avg


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
		default=-1,
	)
	parser.add_argument(
		"--p",
		help='percentile',
		type=str,
		default='p20-sample',
	)
	parser.add_argument(
		"--mode",
		help="data modality, options: {'ca2', 'bold'}",
		type=str,
		choices={'ca2', 'bold'},
		default='ca2',
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
		choices={'pearson', 'spearman', 'cosine'},
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

	# change ulimit
	import resource
	soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))

	# choose ordering and metric used to align
	dmetric = 'cosine'
	ordering = 'infer'
	props = {
		'mice': mice,
		'perc': args.p,
		'num_k': None,
		'mode': args.mode,
		'task': None,
		'metric': args.metric,
		'graph_type': args.graph_type,
		'sv_props': {
			'match_metric': args.match_metric,
			'match_using': args.match_using},
		'dist_metric': dmetric,
		'ordering': ordering,
		'verbose': False,
	}
	# setup variables & pbar
	if args.mode == 'ca2':
		bands = [0.01, 0.5, (0.01, 0.5)]
	else:
		bands = [0.01]
	if args.task == 'all':
		tasks = ['all', 'rest', 'led']
	else:
		tasks = [args.task]
	if args.k == -1:
		num_k = range(3, 26)
	else:
		num_k = [args.k]
	if args.desc == 'all':
		descs = default_descs(args.mode)
	else:
		descs = [args.desc]
	pbar = tqdm(
		itertools.product(tasks, descs, bands, num_k),
		position=0,
		leave=True,
		dynamic_ncols=True,
		total=len(tasks)*len(descs)*len(bands)*len(num_k),
	)
	for task, desc, b, kk in pbar:
		msg = f"running group (task-{task}, desc-{desc}, "
		msg += f"b-{str(b).replace(' ', '')}, kk-{kk})"
		pbar.set_description(msg)
		kws = {
			'task': task,
			f"band_{args.mode.split('-')[0]}": b,
			f"desc_{args.mode.split('-')[0]}": desc,
		}
		mice.setup_func_data(**kws)
		props.update({'num_k': kk, 'task': task})
		gr = Group(**props)
		if gr.svinets_exist():
			gr.fit_group(force=False)

	print('\n')
	print(args)
	print(f"[PROGRESS] analysis.group done {now(True)}.\n")


if __name__ == "__main__":
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		main()
