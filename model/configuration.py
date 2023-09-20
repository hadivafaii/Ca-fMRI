from utils.generic import *


class BaseConfig(object):
	def __init__(
			self,
			main_dir: str,
			download_dir: str = 'path/to/download',
			raw_dir: str = 'raw',
			base_dir: str = 'path/to/base',
			svinet_dir: str = 'Documents/workspaces/svinet',
			makedirs: bool = True,
	):
		super(BaseConfig, self).__init__()

		self.download_dir = download_dir
		self.raw_dir = pjoin(download_dir, raw_dir)
		self.svinet_dir = pjoin(os.environ['HOME'], svinet_dir)
		self.base_dir = pjoin(os.environ['HOME'], base_dir)
		self.stim_dir = pjoin(self.base_dir, 'stim')
		self.processed_dir = pjoin(self.base_dir, 'processed')
		self.tx_dir = pjoin(self.processed_dir, 'transforms')
		self.warped_dir = pjoin(self.processed_dir, 'warped')
		self.main_dir = pjoin(self.processed_dir, main_dir)
		self.ca_dir = pjoin(self.main_dir, 'ca2')
		self.bold_dir = pjoin(self.main_dir, 'bold')
		self.masks_dir = pjoin(self.main_dir, 'masks')
		self.parcel_dir = pjoin(self.main_dir, 'parcellation')
		self.results_dir = pjoin(self.main_dir, 'results')
		self.log_dir = pjoin(self.main_dir, 'log')

		self._get_all_dirs()
		if makedirs:
			self._mkdirs()

	def _get_all_dirs(self):
		dirs = {k: getattr(self, k) for k in dir(self) if '_dir' in k}
		dirs = filter(lambda x: isinstance(x[1], str), dirs.items())
		self.all_dirs = dict(dirs)
		return

	def _mkdirs(self):
		for _dir in self.all_dirs.values():
			os.makedirs(_dir, exist_ok=True)

	def save(self):
		raise NotImplementedError

	def get_main_dir(self):
		raise NotImplementedError


class Config(BaseConfig):
	def __init__(
			self,
			nn: int = 128,
			ll: int = 3,
			resolution: int = 100,
			parcel_mode: str = 'columnar',
			normalization: str = 'global',
			group: str = 'SLC',
			fs_ca: int = 10,
			fs_bold: int = 1,
			exclude: int = 50,
			run_duration: int = 600,
			sub_ids: Iterable[int] = range(1, 10 + 1),
			ses_ids: Iterable[int] = range(1, 3 + 1),
			run_ids: Iterable[int] = range(1, 7 + 1),
			random_state: int = 42,
			**kwargs,
	):
		_allowed_norms = ['global', 'temporal', 'spatial', 'none']
		assert normalization in _allowed_norms,\
			f"allowed normalizations:\n{_allowed_norms}"
		_allowed_parcel_modes = ['columnar', 'spatial', 'allen']
		assert parcel_mode in _allowed_parcel_modes,\
			f"allowed parcellation modes:\n{_allowed_parcel_modes}"
		_allowed_num_layers = [3, 2, 1, 0]  # TODO: later remove 0
		assert ll in _allowed_num_layers,\
			f"allowed num layers:\n{_allowed_num_layers}"
		if parcel_mode == 'spatial':
			ll = 1
		self.nn = nn
		self.ll = ll
		self.resolution = resolution
		self.parcel_mode = parcel_mode
		self.normalization = normalization
		super(Config, self).__init__(self.get_main_dir(), **kwargs)

		_allowed_groups = ['SLC', 'DKI', 'WT']
		assert group in _allowed_groups,\
			f"allowed groups:\n{_allowed_groups}"
		self.group = group

		self.fs_ca = fs_ca
		self.fs_bold = fs_bold
		self.exclude = exclude
		self.run_duration = run_duration

		self.sub_ids = list(sub_ids)
		self.ses_ids = list(ses_ids)
		self.run_ids = list(run_ids)

		self.random_state = random_state
		self.save()

	def save(self):
		file = pjoin(self.main_dir, 'config.json')
		if os.path.isfile(file):
			return
		params = inspect.signature(self.__init__).parameters
		vals = {
			k: getattr(self, k)
			for k, p in params.items()
			if int(p.kind) == 1  # gets rid of args, kwargs
		}
		save_obj(
			obj=vals,
			file_name='config',
			save_dir=self.main_dir,
			mode='json',
		)
		return

	def get_main_dir(self):
		return '_'.join([
			f"norm-{self.normalization}",
			f"parcel-{self.parcel_mode}",
			f"n-{self.nn}"
			if self.parcel_mode == 'spatial'
			else f"n-{self.nn}*{self.ll}",
		])

	def _load_labels(self):
		label_file = pjoin(self.base_dir, 'Yale', 'label_names.rtf')
		with open(label_file, 'r') as f:
			lines = f.readlines()

		start_idx = ['labels' in x for x in lines].index(True)

		labels = {}
		for line in lines[start_idx + 1:-1]:
			idx, lbl = line \
				.replace("\\\'a0 ", '') \
				.replace("\\\'a0", '') \
				.replace('"', '') \
				.split(':')
			idx = int(idx)
			lbl = lbl.replace('\\\n', '').strip()
			if lbl[-1] == ',':
				lbl = lbl[:-1]
			labels[idx] = lbl

		return labels


def regions_to_include(
		allen,
		childs_of: List[str] = None,
		extras: List[str] = None,
		remove: List[str] = None, ):
	childs_of = childs_of if childs_of else [
		'PAL',
		'STR',
		'OLF',
		'HIP',
		'ENT',
		'RHP',
		'MY',
		'P',
		'IB',
		'TH',
		'MB',
		'CB',
		'CBX',
		'HEM',
	]
	extras = extras if extras else [
		'CTXsp',
		'Isocortex',
	]
	remove = remove if remove else [
		'FC',
		'IG',
		'ENTmv',
		'CBXmo',
		'CBXpu',
		'CBXgr',
		'MY-sat',
	]
	structs = allen.s_tree.get_structures_by_acronym(childs_of)
	children = allen.s_tree.children([e['id'] for e in structs])
	children_acros = [e['acronym'] for item in children for e in item]
	regions = [e for e in children_acros if e not in childs_of + remove]
	regions += extras
	return regions
