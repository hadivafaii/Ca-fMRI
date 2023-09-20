from .group import *


class BootStrap(Group):
	def __init__(
			self,
			batch: int = 5,
			n_jobs: int = -1,
			n_resamples: int = int(1e4),
			est: Callable = bn.nanmean,
			**kwargs
	):
		super(BootStrap, self).__init__(**kwargs)
		self.n_resamples = n_resamples
		self.n_jobs = n_jobs
		self.batch = batch
		self.est = est
		self.kws_job = {
			'backend': 'loky',
			'inner_max_num_threads': 1
			if n_jobs == -1 else 4,
		}
		self.seed_h = None
		self.run_h = None

	def fit_boot(
			self,
			indiv: bool = False,
			use_scipy: bool = True,
			n_resamples: int = None,
			force: bool = False, ):
		if n_resamples is not None:
			self.n_resamples = n_resamples
		se, fname = self.load_boot(indiv, use_scipy)
		if not force:
			if se is None:
				if self.verbose:
					print('Not found . . . initiating fit')
			else:
				return se, fname

		if use_scipy:
			if indiv:
				se = self._fit_indiv_scipy()
			else:
				fn = self.get_booter(use_scipy=True)
				se = fn(bn.nanmean(self.pi_prune, -3))
		else:
			se = self._fit_simple(indiv)
		save_obj(
			obj=se,
			file_name=fname,
			save_dir=self.path(),
			verbose=self.verbose,
			mode='npy',
		)
		return se, fname

	def _fit_indiv_scipy(self, reshape_runs: bool = True):
		n_runs = self.get_anim_num_runs()
		data_all, good_nodes = {}, {}
		for i in range(self.mice.cfg.n_subjects):
			x = self.run_pi[i]
			if reshape_runs:
				x = x.reshape((-1, *x.shape[-3:]))

				nans = np.isnan(x).sum(-1).sum(-1)
				nans = nans == np.prod(x.shape[-2:])
				good_runs = np.where(nans.sum(-1) < nans.shape[-1])[0]
				assert len(good_runs) == n_runs[self.mice.get_key(i + 1)]

				x = x[good_runs]
				nans = nans[good_runs]
				good_seeds = np.where(nans.sum(0) < len(good_runs))[0]
				if len(good_seeds) < x.shape[-3]:
					x = x[:, good_seeds]

				nans = np.isnan(x).sum(0).sum(0).sum(0)
				good = nans < np.prod(x.shape[:3])
				data_all[i] = x[..., good]
				good_nodes[i] = good

			else:
				good = np.isnan(avg(x)).sum(0) == 0
				good_nodes[i] = good
				x = x[..., good]

				nans = np.isnan(x).sum(-1).sum(-1) == np.prod(x.shape[-2:])
				nans = nans.reshape(-1, nans.shape[-1])
				good_seeds = nans.sum(0) < np.prod(x.shape[:2]) // 2
				if good_seeds.sum() < x.shape[-3]:
					x = x[:, :, good_seeds]
				data_all[i] = x

		fn = self.get_booter(use_scipy=True)
		with joblib.parallel_backend(**self.kws_job):
			se_list = joblib.Parallel(n_jobs=self.n_jobs)(
				joblib.delayed(fn)(x)
				for x in data_all.values()
			)

		# fix degenerate nodes
		data_tmp = [
			self.run_pi[i][..., good] for
			i, good in good_nodes.items()
		]
		with joblib.parallel_backend(**self.kws_job):
			se_list = joblib.Parallel(n_jobs=self.n_jobs)(
				joblib.delayed(self._final_indiv_fix)(x, se)
				for x, se in zip(data_tmp, se_list)
			)

		# concatenate together
		se_indiv = np_nans((
			self.mice.cfg.n_subjects,
			self.num_k,
			self.num_n,
		))
		for i, good in good_nodes.items():
			se_indiv[i][:, good] = se_list[i]
		return se_indiv

	def _final_indiv_fix(self, x, se):
		nans = np.isnan(se).sum(0) == len(se)
		if nans.sum():
			b = bootstrap(
				x=x[..., nans],
				batch=self.batch,
				n_resamples=self.n_resamples,
				statistic=lambda a: avg(a, 2, 1),
				dims=3,
			)
			se[..., nans] = bn.nanstd(b, 0, ddof=1)
		return se

	def _fit_simple(self, indiv):
		fn = self.get_booter(use_scipy=False)
		if indiv:
			with joblib.parallel_backend(**self.kws_job):
				se_list = joblib.Parallel(n_jobs=self.n_jobs)(
					joblib.delayed(fn)(x)
					for x in self.run_pi
				)
			se = [np.expand_dims(x, 0) for x in se_list]
			se = np.concatenate(se)
		else:
			se = fn(bn.nanmean(self.pi_prune, -3))
		return se

	def load_boot(self, indiv: bool, use_scipy: bool):
		file, fname = self.get_boot_name(indiv, use_scipy)
		if os.path.isfile(file):
			return np.load(file), fname
		else:
			return None, fname

	def get_boot_name(self, indiv: bool, use_scipy: bool):
		f = [
			'se-BCa' if use_scipy else 'se',
			f"{'indiv' if indiv else 'group'}-pi",
			f"{self.n_resamples:1.1e}",
		]
		fname = '_'.join(f)
		file = pjoin(self.path(), f"{fname}.npy")
		return file, fname

	def get_booter(
			self,
			n_resamples: int = None,
			use_scipy: bool = True, ):
		n_resamples = n_resamples if n_resamples else self.n_resamples
		if use_scipy:
			def do_boot(x, statistic=None):
				if statistic is None:
					statistic = mk_statistic(x.ndim - 3)
				return sp_stats.bootstrap(
					data=(x,),
					statistic=statistic,
					random_state=self.mice.cfg.random_state,
					n_resamples=n_resamples,
					batch=self.batch,
					vectorized=True,
					method='BCa',
					axis=0,
				).standard_error
		else:
			def do_boot(x, statistic=None):
				if statistic is None:
					def statistic(a):
						return avg(a, 2, 1)
				b = bootstrap(
					x=x,
					batch=self.batch,
					n_resamples=n_resamples,
					statistic=statistic,
					dims=x.ndim - 2,
				)
				return bn.nanstd(b, 0, ddof=1)
		return do_boot

	def prune(
			self,
			alpha: float = 0.05,
			thres: float = 0.003,
			use_scipy: bool = True,
			n_resamples: int = None,
			save: bool = True, ):
		se, _ = self.fit_boot(
			indiv=True,
			use_scipy=use_scipy,
			n_resamples=n_resamples,
			force=False,
		)
		reject, _ = self.ttest(
			se=se,
			mu=thres,
			alpha=alpha,
		)
		reject = np.expand_dims(np.expand_dims(np.expand_dims(
			reject, 1), 1), 1)
		for i in range(1, 4):
			reject = np.repeat(reject, self.run_pi.shape[i], axis=i)
		pi_prune = self.run_pi.copy()
		pi_prune[reject == 0] = 0.0
		pi_prune /= np.nansum(pi_prune, -2, keepdims=True)
		pi_prune, _ = self.align_post_prune(pi_prune, save)
		return pi_prune

	def ent_deg_percentile(
			self,
			n_resamples: int = int(1e5),
			mu: Tuple[float, float] = None,
			good: np.ndarray = None, ):
		if self.run_h is not None:
			ent = self.run_h
		else:
			pi = self.pi_prune
			pi = bn.nanmean(pi, -3)
			ent = sp_stats.entropy(pi, axis=-2)
			ent /= np.log(pi.shape[-2])

		x = np.concatenate([
			np.expand_dims(ent, -2),
			np.expand_dims(self.run_deg, -2),
		], axis=-2)
		if good is not None:
			x = x[..., good]
		se = self.get_booter(n_resamples)(x)
		se_tmp = self.get_booter(n_resamples, False)(x)
		se[np.isnan(se)] = se_tmp[np.isnan(se)]

		if mu is None:
			mu = tuple(
				bn.nanmedian(x[..., i, :])
				for i in range(2)
			)
		tvals = np_nans((2, x.shape[-1]))
		percentiles = np_nans((2, x.shape[-1]))
		for i in range(2):
			t = (avg(x[..., i, :], 1) - mu[i]) / se[i]
			tvals[i] = t
			percentiles[i] = [
				sp_stats.percentileofscore(
					a=t[np.isfinite(t)],
					score=e,
				) for e in t
			]
		output = {
			'ent': x[..., 0, :],
			'deg': x[..., 1, :],
			'se_ent': se[0],
			'se_deg': se[1],
			't_ent': tvals[0],
			't_deg': tvals[1],
			'p_ent': percentiles[0],
			'p_deg': percentiles[1],
		}
		if good is not None:
			_output = {}
			for k, x in output.items():
				shape = x.shape[:-1]
				shape += (self.num_n,)
				y = np_nans(shape)
				y[..., good] = x
				_output[k] = y
			return _output
		else:
			return output

	def compute_entropy(
			self,
			use_prune: bool = True,
			select_seeds: bool = False, ):
		# Seeds
		pi = self.pi_prune if use_prune else self.run_pi
		h = sp_stats.entropy(pi, axis=-2)
		h /= np.log(pi.shape[-2])
		self.seed_h = h

		# Runs
		n_seeds = self.pi_prune.shape[-3]
		scores = np.zeros(n_seeds)
		if select_seeds:
			for i in range(0, n_seeds):
				pi = self._get_pi_seed_idx(i, use_prune)
				h = sp_stats.entropy(pi, axis=-2) / np.log(pi.shape[-2])
				scores[i] = self.symmetry_score(
					pi=avg(h, 1), decimals=10)[0][-1]
			i = np.nanargmax(scores)
		else:
			i = n_seeds
		# use selected i to compute h
		pi = self._get_pi_seed_idx(i, use_prune)
		h = sp_stats.entropy(pi, axis=-2)
		h /= np.log(pi.shape[-2])
		self.run_h = h
		self._add_entropy_ranks()
		self._add_degree_ranks()
		return scores, i

	def _get_pi_seed_idx(self, i, use_prune):
		if i == 0:
			if use_prune:
				pi = self.pi_prune[..., i, :, :]
			else:
				pi = self.run_pi[..., i, :, :]
		elif i > 0:
			if use_prune:
				pi = self.pi_prune[..., range(0, i), :, :]
			else:
				pi = self.run_pi[..., range(0, i), :, :]
			pi = bn.nanmean(pi, -3)
		else:
			raise ValueError(f"invalid idx: '{i}'")
		return pi

	def _add_entropy_ranks(self, normalize: bool = True):
		hr = np_nans(self.run_h.shape)
		i, j, k = self.run_h.shape[:3]
		looper = itertools.product(
			range(i), range(j), range(k))
		for i, j, k in looper:
			x = self.run_h[i, j, k]
			good = np.isfinite(x)
			if not good.sum():
				continue
			x = sp_stats.rankdata(x[good])
			if normalize:
				x /= np.nanmax(x)
			hr[i, j, k][good] = x
		self.run_hr = hr
		return

	def _add_degree_ranks(self, normalize: bool = True):
		dr = np_nans(self.run_deg.shape)
		i, j, k = self.run_deg.shape[:3]
		looper = itertools.product(
			range(i), range(j), range(k))
		for i, j, k in looper:
			x = self.run_deg[i, j, k]
			good = np.isfinite(x)
			if not good.sum():
				continue
			x = sp_stats.rankdata(x[good])
			if normalize:
				x /= np.nanmax(x)
			dr[i, j, k][good] = x
		self.run_dr = dr
		return

	def mk_cartography_df(
			self,
			pi: np.ndarray = None,
			affil: List[str] = None,
			n_resamples: int = int(1e5),
			mu: Tuple[float, float] = None,
			sensory_ocs: List[int] = None,
			good: np.ndarray = None, ):
		if pi is None:
			pi = bn.nanmean(self.pi_prune, -3)
		partition = self.get_disjoint_community(avg(pi))
		affil = affil if affil else self.node_affiliations()[1]

		sensory_ocs = sensory_ocs if sensory_ocs else [2, 3]
		community_type = [
			'sensory' if c in sensory_ocs else
			'association' for c in partition
		]
		results = self.ent_deg_percentile(
			n_resamples=n_resamples, mu=mu, good=good)
		results = {
			k: v if v.ndim == 1 else avg(v, 1)
			for k, v in results.items()
		}
		# low_ent = results['ent'] < bn.nanmedian(results['ent'])
		# low_deg = results['deg'] < bn.nanmedian(results['deg'])
		# tag = [' / '.join([
		# 	f"{'L-deg' if h else 'H-deg'}",
		# 	f"{'L-ent' if d else 'H-ent'}",
		# ]) for h, d in zip(low_ent, low_deg)]

		df_carto = pd.DataFrame({
			'affil': affil,
			'oc_idx': partition,
			'oc_type': community_type,
			'node': self.n2id.keys(),
			**results,
		})
		self._add_mode_df(df_carto)
		return df_carto


def mk_statistic(
		dims: int = 3,
		est: Callable = bn.nanmean, ) -> Callable:
	"""
	Assumes default input is pi to be used with SciPy boot:
		All results arranged in one 7-dim tensor. Has shape
		N_sub x N_ses x N_run x N_seed x K x N, where
		- N_sub = number of subjects
		- N_ses = number of sessions
		- N_run = number of runs
		- N_seed = number of seeds
		- K = number of communities
		- N = number of nodes (ROIs)
	:param dims: number of dimensions to average-out
	:param est: estimator function
	:return: statistic function

	[INFO] SciPy bootstrap procedure:
		- Let's say n_resample = batch * n, and axis = 0
		1. It first resamples data n times along axis
			e.g. x will have shape:
			(n_ses, n_run, n_seed, K, N, batch, n_sub),
			where batch dimension contains resampled
			indices between 0 and 9 (with replacement)
		2. After all n_resamples are done, it takes the real data
			e.g. for this stage x has shape:
			(n_ses, n_run, n_seed, K, N, n_sub), which is just
			the actual data but the first dimension is transposed
			to the last dimension (remember axis = 0).
		3. Finally, we have a batch of size n_sub.  Here for each
			batch a single animal is omitted. e.g. x has shape:
			(n_ses, n_run, n_seed, K, N, n_sub, n_sub - 1)
			where for batch b, we have data from all samples
			minus one.  That is: {0, ..., n_sub - 1} - {b}
		- Note that in the stage 3 the last two dims
			have the same meaning as stage 1:
			it's (..., batch, n_sub), but the difference is
			for each batch, one of the samples is omitted.
	"""
	def statistic(x, axis):
		x = avg(x, x.ndim - dims, 0, est)
		x = est(x, axis=axis)
		return x
	return statistic


def bootstrap(
		x: np.ndarray,
		n_resamples: int,
		batch: int = 5,
		skip: List[str] = None,
		dims: Union[int, dict] = 4,
		statistic: Callable = None,
		rng=None, ):

	if rng is not None:
		assert n_resamples == batch

	if isinstance(dims, int):
		dims = {
			f"d{i}": d for i, d in
			enumerate(x.shape[:dims])
		}
	assert all(
		i == d for i, d in
		zip(x.shape, dims.values())
	), "must provide compatible dims"
	if skip:
		assert all(
			e in dims
			for e in skip
		), "skip dims must be valid"
	booted = []
	for i in range(int(np.ceil(n_resamples / batch))):
		ids, shape = resample(
			dims=dims,
			batch=batch,
			skip=skip,
			rng=rng if rng else i
		)
		ids = tuple(e for e in ids.values())
		shape = (*shape, *x.shape[len(shape) - 1:])
		b = x[ids].reshape(shape)
		if statistic is not None:
			b = statistic(b)
		booted.append(b)
	booted = np.concatenate(booted)
	return booted


def resample(
		dims: Dict[str, int],
		skip: List[str] = None,
		batch: int = 5,
		rng=None, ):
	skip = skip if skip else []
	if isinstance(rng, int):
		rng = get_rng(rng)
	rng = rng if rng else get_rng()

	ids = {}
	sz = batch
	for name, d in dims.items():
		sz *= d
		if name in skip:
			sample = np.tile(
				range(d), sz // d)
		else:
			sample = rng.choice(
				a=d,
				size=sz,
				replace=True,
			)
		ids[name] = sample

	for i, (name, sample) in enumerate(ids.items()):
		repeats = np.array(list(dims.values()))
		repeats = np.prod(repeats[i + 1:])
		if repeats > 1:
			ids[name] = np.repeat(sample, repeats)
	return ids, (batch, *dims.values())


def mk_statistic_fn(
		bins: np.ndarray,
		scipy: bool = False,
		individual: bool = False,
		est_sub: Callable = bn.nanmean,
		est_ses: Callable = bn.nanmean,
		est_run: Callable = bn.nanmean,
		est_seed: Callable = bn.nanmean, ) -> Callable:

	def _perform_avg(x, axis_run, axis_ses):
		return est_ses(est_run(x, axis_run), axis_ses)

	def _resample(x, rng):
		assert x.ndim == 7
		if individual:
			sub, run, seed, k, n, b, ses = x.shape
			dims = {
				'sub': sub,
				'ses': ses,
				'run': run,
				'seed': seed,
			}
			x_resample = np.empty_like(x)
			for i in range(b):
				ids, shape = resample(
					batch=1,
					dims=dims,
					skip=['sub', 'ses'],
					rng=rng,
				)
				ids = tuple(e for e in ids.values())
				shape = (*shape[1:], k, n)
				x_resample[..., i, :] = x.transpose(
					(0, -1, *range(1, x.ndim - 1))
				)[..., i][ids].reshape(shape).transpose(
					(0, *range(2, x.ndim - 1), 1)
				)
		else:
			ses, run, seed, k, n, b, sub = x.shape
			dims = {
				'sub': sub,
				'ses': ses,
				'run': run,
				'seed': seed,
			}
			x_resample = np.empty_like(x)
			for i in range(b):
				ids, shape = resample(
					batch=1,
					dims=dims,
					skip=['sub'],
					rng=rng,
				)
				ids = tuple(e for e in ids.values())
				shape = (*shape[1:], k, n)
				x_resample[..., i, :] = x.transpose(
					(-1, *range(x.ndim - 1))
				)[..., i][ids].reshape(shape).transpose(
					(*range(1, x.ndim - 1), 0)
				)
		return x_resample

	if scipy:
		def _statistic(x, axis=None):
			skip_resample = (
					x.ndim == 6 or
					x.shape[-2] - x.shape[-1] == 1
			)
			if not skip_resample:
				x = _resample(
					x, get_rng(random.randint(0, int(1e6))))

			pi = pi_from_gam(x, 2, 2, est_seed)
			shape = list(pi.shape)
			d1, d2, k, n = shape[:4]
			shape[3] = len(bins) - 1
			cut = np_nans(shape)
			for i in range(d1):
				for j in range(d2):
					_cut, _ = cut_data(
						data=pi[i, j].reshape(k, n, -1),
						bins=bins,
						axis_oc=0,
						axis_node=1,
						axis_batch=-1,
						oc_only=True,
					)
					cut[i, j] = _cut.reshape(*shape[2:])

			if individual:
				cut = est_run(cut, 1)
				cut = est_ses(cut, axis)
			else:
				cut = _perform_avg(cut, 1, 0)
				cut = est_sub(cut, axis)

			return cut
		return _statistic

	else:
		def _statistic(x):
			pi = pi_from_gam(x)
			k, n = pi.shape[-2:]
			cut, _ = cut_data(
				data=pi.reshape(-1, k, n),
				bins=bins,
				oc_only=True,
			)
			cut = cut.reshape((*pi.shape[:-2], k, -1))

			pi = _perform_avg(pi, -3, -3)
			cut = _perform_avg(cut, -3, -3)

			output = {
				'pi': pi,
				'cut': cut,
				'pi_avg': est_sub(pi, -3),
				'cut_avg': est_sub(cut, -3),
			}
			return output
		return _statistic


def corr_group(
		x: np.ndarray,
		y: np.ndarray,
		n_resamples: int = int(1e5), ):
	assert x.ndim == y.ndim
	b_1 = bootstrap(
		x=x,
		n_resamples=n_resamples,
		statistic=lambda a: avg(a, 1, 1),
		dims=x.ndim - 1,
	)
	b_2 = bootstrap(
		x=y,
		n_resamples=n_resamples,
		statistic=lambda a: avg(a, 1, 1),
		dims=y.ndim - 1,
	)
	data = np.concatenate([
		np.expand_dims(b_1, 1),
		np.expand_dims(b_2, 1),
	], axis=1)

	corrs = []
	for ii in range(n_resamples):
		good = np.isnan(data[ii]).sum(0) == 0
		d = sp_dist.pdist(
			X=data[ii][:, good],
			metric='correlation',
		)
		corrs.append(dist2sim(d))
	se = bn.nanstd(corrs, ddof=1)

	x = avg(x, 1)
	y = avg(y, 1)
	good = np.logical_and(
		np.isfinite(x),
		np.isfinite(y),
	)
	rho = sp_stats.pearsonr(x[good], y[good])
	return rho, se


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
		"--num",
		help='num resamples',
		type=int,
		default=1000,
	)
	parser.add_argument(
		"--njobs",
		help='number of jobs',
		type=int,
		default=-1,
	)
	parser.add_argument(
		"--mode",
		help="data modality, options: {'ca2', 'bold'}",
		type=str,
		choices={'ca2', 'bold'},
		default='ca2',
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
	parser.add_argument(
		"--scipy",
		help='whether to use scipy BCa method',
		action='store_true',
		default=True,
	)
	parser.add_argument(
		"--group",
		help='follow up with group boot?',
		action='store_true',
		default=True,
	)
	return parser.parse_args()


def main():
	args = _setup_args()
	mice = Mice(Config(
		nn=args.nn,
		ll=args.ll,
		parcel_mode=args.parcel_mode,
	))
	props = {
		'mice': mice,
		'perc': args.p,
		'num_k': None,
		'mode': args.mode,
		'task': None,
		'metric': args.metric,
		'sv_props': {
			'match_metric': args.match_metric,
			'match_using': args.match_using},
		'graph_type': args.graph_type,
		'verbose': False,
	}
	if args.mode == 'ca2':
		bands = [(0.01, 0.5), 0.5]
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

	prune_alpha = 0.05
	prune_thres = 0.003

	looper = itertools.product(tasks, bands, num_k)
	pbar = tqdm(
		looper,
		position=0,
		leave=True,
		dynamic_ncols=True,
		total=len(tasks) * len(bands) * len(num_k),
	)
	for task, b, kk in pbar:
		msg = f"running boot (task-{task}, "
		msg += f"b-{str(b).replace(' ', '')}, kk-{kk})"
		pbar.set_description(msg)

		kws = {
			'task': task,
			f"band_{args.mode}": b,
		}
		mice.setup_func_data(**kws)
		props.update({'num_k': kk, 'task': task})

		boot = BootStrap(
			batch=5,
			n_jobs=mice.cfg.n_subjects,
			n_resamples=args.num,
			est=bn.nanmean,
			**props,
		)
		if not hasattr(boot, 'run_pi'):
			continue
		_ = boot.fit_boot(
			indiv=True,
			use_scipy=args.scipy,
			n_resamples=args.num,
			force=False,
		)
		if boot.pi_prune is None:
			boot.pi_prune = boot.prune(
				alpha=prune_alpha,
				thres=prune_thres,
				use_scipy=args.scipy,
				save=True,
			)
		if args.group:
			_ = boot.fit_boot(
				indiv=False,
				use_scipy=args.scipy,
				n_resamples=int(1e5),
				force=False,
			)
		# TODO: this has to be a separate loop
		#  (because args.num could be allowed to be different)

	print('\n')
	print(args)
	print(f"[PROGRESS] analysis.bootstrap done {now(True)}.\n")


if __name__ == "__main__":
	# main()
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		main()
