import os.path

from utils.generic import *
from .configuration import Config
from .dataset import AnatData, FuncData

from register.atlas import Allen
from register.parcellation import Parcellation


BOLD = 'BOLD'
CAS = r'Ca$^{2\!\!+}_{slow}$'
CAF = r'Ca$^{2\!\!+}_{fast}$'

_EXCLUDE_CA = [
	(2, 1, 6),
	(5, 2, 3),
	(7, 2, 2),  # QC
	(5, 1, 5),  # QC
]
_EXCLUDE_BOLD = []


class Mice(object):
	def __init__(
			self,
			cfg: Union[int, Tuple[int, ...], Config] = None,
			full_load: bool = False,
			load_allen: bool = False,
			load_parcel: bool = False,
			**kwargs,
	):
		super(Mice, self).__init__()
		if isinstance(cfg, int):
			self.cfg = Config(cfg)
		elif isinstance(cfg, tuple):
			self.cfg = Config(*cfg)
		else:
			self.cfg = cfg if cfg else Config()
		self.al = Allen(
			resolution=self.cfg.resolution,
			full_load=load_allen,
		)
		self.parcel = Parcellation(
			cfg=self.cfg,
			allen=self.al,
			full_load=load_parcel,
		)
		self.looper = self._setup_looper()
		self.looper_agg = {
			k: ssr for k, ssr in
			self.looper.items()
			if ssr[-1] == -1
		}
		self.ca, self.bold = {}, {}
		self._load_node_lookup()
		self.mismatch = []

		self.kws = {}
		self.band = {}
		self.set_kws(**kwargs)

		self.ref = {}
		self.T1w = {}
		self.mask2d = {}
		self.mask3d = {}
		self.ca_preproc = {}
		self.bold_preproc = {}
		if full_load:
			self.setup_anat_data()
			self.setup_preproc_data()
			self.setup_func_data()

	def get_key(
			self,
			sub: int = -1,
			ses: int = -1,
			run: int = -1, ):
		try:
			key = next(
				k for k, v in
				self.looper.items()
				if v == (sub, ses, run)
			)
		except StopIteration:
			key = None
		return key

	def get_condition_ids(self, keys: Iterable[str] = None):
		keys = keys if keys else [
			key for key, (_, _, run) in
			self.looper.items() if run != -1
		]
		run_ids = {}
		for sub in self.cfg.sub_ids:
			for ses in self.cfg.ses_ids:
				_k = self.get_key(sub, ses)
				run_ids[_k] = [
					i for i, k in enumerate(keys)
					if _k in k
				]

		within_ids = collections.defaultdict(list)
		for _k, ids in run_ids.items():
			for i, j in itertools.combinations(range(len(ids)), 2):
				within_ids[_k].append((ids[i], ids[j]))

		across_ids = collections.defaultdict(list)
		for sub in self.cfg.sub_ids:
			for a, b in itertools.combinations(self.cfg.ses_ids, 2):
				ka = self.get_key(sub, a)
				kb = self.get_key(sub, b)
				ids_a = run_ids[ka]
				ids_b = run_ids[kb]
				_k = f"sub{sub}_ses{a}v{b}"
				across_ids[_k] = list(itertools.product(ids_a, ids_b))

		ids = {
			'run': run_ids,
			'within': dict(within_ids),
			'across': dict(across_ids),
		}
		return ids

	def combine_keys(self, mode: str):
		combine = collections.defaultdict(list)
		proc, *_ = self.get_data_containers(mode)
		for key in self.looper_agg:
			combine[key] = sorted(filter(
				lambda k: all(
					e in k for e in
					key.split('_')),
				list(proc),
			))
		return dict(combine)

	def cat_data(self, **kwargs):  # TODO: Useless, delete
		self.set_kws(**kwargs)
		looper = {
			'ca2': self.cfg.ca_dir,
			'bold': self.cfg.bold_dir,
		}
		combine_info = collections.defaultdict(dict)
		for mode, save_dir in looper.items():
			for task in ['rest', 'led', 'all']:
				self.setup_func_data(task=task)
				combined, meta = self._concatenate(mode, task)
				combine_info[f"{mode}_{task}"] = meta
				for key, x in combined.items():
					file_name = _get_cat_name(
						key=key,
						task=task,
						mode=mode,
					)
					save_obj(
						obj=x,
						file_name=file_name,
						save_dir=save_dir,
						verbose=False,
						mode='npy',
					)
		save_obj(
			obj=dict(combine_info),
			file_name='combine_info',
			save_dir=self.cfg.main_dir,
			verbose=True,
			mode='npy',
		)
		return

	def _concatenate(self, mode: str, task: str):
		assert task in ['rest', 'led', 'all'], \
			f"invalid task: '{task}'"
		tasks = ['rest', 'led'] if task == 'all' else [task]

		combined, meta = {}, {}
		for key in self.looper_agg:
			pooled = self._pool_data(key, mode, tasks)
			if len(pooled):
				x = [e for e in pooled.values()]
				combined[key] = np.concatenate(x, axis=1)
				meta[key] = list(pooled)
		for ses in self.cfg.ses_ids:
			pooled = self._pool_data(f"ses-{ses}", mode, tasks)
			assert all(self.looper[k][1] == ses for k in pooled)
			key = self.get_key(ses=ses)
			x = [e for e in pooled.values()]
			combined[key] = np.concatenate(x, axis=1)
			meta[key] = list(pooled)
		return combined, meta

	def _pool_data(self, key, mode, tasks, verbose=True):
		proc, *_ = self.get_data_containers(mode)
		data = {}
		for k, func in proc.items():
			cond = func.task in tasks
			cond = cond and key in k
			if not cond:
				continue
			x = func[:]
			nans = np.isnan(x).sum(1)
			nans = nans.astype(bool).sum()
			nans /= len(x)
			if nans > 0.5:
				msg = f"Warning, {k}: {100 * nans:0.0f}"
				msg += ' % of ROIs are nan. '
				msg += 'Moving on . . .'
				if verbose:
					print(msg)
				continue
			data[k] = x
		return data

	def load_stim_times(self, mode: str):
		proc, _, _, fs, stim_key = self.get_data_containers(mode)
		dlist = []
		for key, ssr in self.looper.items():
			if key not in proc or any(i == -1 for i in ssr):
				continue
			a = self.cfg.exclude
			a += self.cfg.run_duration * (ssr[-1] - 1)
			b = self.cfg.run_duration * ssr[-1]
			timepoint = range(a * fs, b * fs)
			try:
				f = os.listdir(self.cfg.stim_dir)
				f = next(x for x in sorted(f) if key in x)
				f = pjoin(self.cfg.stim_dir, f)
				stim = pd.read_csv(f, index_col=0)
				stim = stim.loc[:, stim_key]
				stim = stim.dropna().values
			except StopIteration:
				stim = np.zeros(self.cfg.run_duration * fs)
			dlist.append({
				'key': [key] * len(timepoint),
				'mode': [mode] * len(timepoint),
				'task': [proc[key].task] * len(timepoint),
				'timepoint': timepoint,
				'stim': stim[-len(timepoint):].astype(int),
			})
		return pd.DataFrame(merge_dicts(dlist))

	def setup_anat_data(self):
		for key, (sub, ses, run) in self.looper.items():
			if sub == -1 or ses == -1 or run != -1:
				continue
			self.ref[key] = AnatData(key, 'referenceimage_ca2', self.cfg)
			self.T1w[key] = AnatData(key, 'T1w', self.cfg)
			self.mask2d[key] = AnatData(key, 'mask_ca2', self.cfg)
			self.mask3d[key] = AnatData(key, 'mask_bold', self.cfg)
		return

	def setup_preproc_data(self, **kwargs):
		self.set_kws(**kwargs)
		props = {
			'cfg': self.cfg,
			'space': 'individual',
		}
		for key, (sub, ses, run) in self.looper.items():
			if any(i == -1 for i in [sub, ses, run]):
				continue
			self.ca_preproc[key] = FuncData(
				key, 'ca2', desc=self.kws.get('desc_ca2'), **props)
			self.bold_preproc[key] = FuncData(
				key, 'bold', desc=self.kws.get('desc_bold'), **props)
		return

	def set_kws(self, **kwargs):
		if not self.kws:
			defaults = dict(
				task='rest',
				exclude=True,
				runs_only=True,
				band_ca2=_setup_band(
					0.01, self.cfg.fs_ca),
				band_bold=_setup_band(
					0.01, self.cfg.fs_bold),
				desc_ca2='preproc',
				desc_bold='rabies-hp',
			)
			self.kws = setup_kwargs(
				defaults=defaults,
				kwargs=kwargs,
			)
		for k, v in kwargs.items():
			if k == 'band_ca2':
				self.kws[k] = _setup_band(
					v, self.cfg.fs_ca)
			elif k == 'band_bold':
				self.kws[k] = _setup_band(
					v, self.cfg.fs_bold)
			else:
				self.kws[k] = v
		assert all(
			isinstance(self.kws[e], str) for e in
			['task', 'desc_ca2', 'desc_bold']
		)
		return

	def setup_func_data(self, **kwargs):
		self.set_kws(**kwargs)
		props = {
			'space': 'CCF',
			'cfg': self.cfg,
			'task': self.kws['task'] if
			self.kws['task'] in ['rest', 'led']
			else None,
		}
		self.ca, self.bold = {}, {}
		for key in self.looper:
			if self.kws['runs_only'] and 'run' not in key:
				continue
			# ca
			func = FuncData(
				key=key,
				mode='ca2',
				desc=self.kws.get('desc_ca2'),
				**props,
			)
			if self.kws['exclude'] and self.looper[key] in _EXCLUDE_CA:
				skip = True
			else:
				skip = func.npy_file is None
			if not skip:
				self.ca[key] = func
			# bold
			func = FuncData(
				key=key,
				mode='bold',
				desc=self.kws.get('desc_bold'),
				**props,
			)
			if self.kws['exclude'] and self.looper[key] in _EXCLUDE_BOLD:
				skip = True
			else:
				skip = func.npy_file is None
			if not skip:
				self.bold[key] = func
		return

	def get_data_containers(
			self,
			mode: str,
			force: bool = False,
			**kws_func, ):
		def _retrieve():
			if mode == 'ca2':
				_proc = getattr(self, 'ca', {})
				_preproc = getattr(self, 'ca_preproc', {})
				_masks = getattr(self, 'mask2d', {})
				_fs = self.cfg.fs_ca
				_stim_key = 'ledStim10Hz'
			elif mode == 'bold':
				_proc = getattr(self, 'bold', {})
				_preproc = getattr(self, 'bold_preproc', {})
				_masks = getattr(self, 'mask3d', {})
				_fs = self.cfg.fs_bold
				_stim_key = 'ledStim1Hz'
			else:
				raise ValueError(f"invalid mode: '{mode}'")
			return _proc, _preproc, _masks, _fs, _stim_key
		if force or not (self.ca or self.bold):
			self.setup_func_data(**kws_func)
		return _retrieve()

	def _load_node_lookup(self):
		lookup = 'roi_lookup.npy'
		lookup = pjoin(self.cfg.main_dir, lookup)
		try:
			node_lookup = np.load(
				lookup, allow_pickle=True).item()
			# noinspection PyTypeChecker
			self.node_lookup = dict(node_lookup)
		except FileNotFoundError:
			msg = 'ROI lookup not found, '
			msg += 'time to run organize_func() . . .'
			print(msg)
			self.node_lookup = {}
		return

	def _setup_looper(self):
		looper = {}

		# across runs
		for sub in self.cfg.sub_ids:
			for ses in self.cfg.ses_ids:
				for run in self.cfg.run_ids:
					key = [
						f"sub-{self.cfg.group}{sub:02d}",
						f"ses-{ses:d}",
						f"run-{run:d}",
					]
					key = '_'.join(key)
					val = (sub, ses, run)
					looper[key] = val

		# across sessions (cat all runs)
		for sub in self.cfg.sub_ids:
			for ses in self.cfg.ses_ids:
				key = [
					f"sub-{self.cfg.group}{sub:02d}",
					f"ses-{ses:d}",
				]
				key = '_'.join(key)
				if not any(key in e for e in looper):
					continue
				val = (sub, ses, -1)
				looper[key] = val

		# across subjects (cat all sessions and runs)
		for sub in self.cfg.sub_ids:
			key = f"sub-{self.cfg.group}{sub:02d}"
			val = (sub, -1, -1)
			looper[key] = val

		# global (cat all subjects, sessions, and runs)
		key = f"sub-{self.cfg.group}"
		val = (-1, -1, -1)
		looper[key] = val

		# across sessions (cat all subjects)
		for ses in self.cfg.ses_ids:
			key = f"sub-{self.cfg.group}_ses-{ses:d}"
			val = (-1, ses, -1)
			looper[key] = val
		return looper


def _setup_band(b, fs):
	if isinstance(b, tuple):
		assert 0 <= b[0] and b[1] <= fs / 2
		return b
	elif isinstance(b, (float, int)):
		assert 0 <= b < fs / 2
		return float(b), fs / 2
	elif b is None:
		return 0.0, fs / 2
	else:
		raise RuntimeError(f"invalid band: '{b}'")


def _get_cat_name(key, task, mode):
	dim = '2d' if mode == 'ca2' else '3d'
	name = [
		key,
		f"task-{task}",
		f"space-CCF{dim}",
		mode,
	]
	return '_'.join(name)
