from utils.generic import *
from .configuration import Config
import fnmatch


class ImagingData(object):
	def __init__(
			self,
			key: str,
			mode: str,
			space: str,
			cfg: Config,
			patterns: List[str] = None,
	):
		super(ImagingData, self).__init__()
		self.cfg = cfg if cfg else Config()

		_allowed_spaces = ['individual', 'CCF']
		assert space in _allowed_spaces,\
			f"allowed spaces:\n{_allowed_spaces}"

		patterns = patterns if patterns else []
		_patterns = patterns + [
			f"space-{space}",
			*key.split('_'),
			mode,
		]
		self._patterns = sorted(set(_patterns))

		self.key = key
		self.mode = mode
		self.space = space

		if self.space == 'individual':
			self.npy_dir = None
		else:
			if self.mode == 'bold':
				self.npy_dir = self.cfg.bold_dir
			elif self.mode == 'ca2':
				self.npy_dir = self.cfg.ca_dir
			elif self.mode in ['atlas2d-inv', 'atlas3d-inv']:
				self.npy_dir = self.cfg.masks_dir
			else:
				msg = "invalid combo:\n"
				msg += f"mode = '{self.mode}',  "
				msg += f"space = '{self.space}'"
				raise ValueError(msg)

		self.nii_file = None
		self.npy_file = None
		self._get_files()

		self.dtype = self._get_dtype()
		self._npy = self._load_npy()
		try:
			self.shape = self._npy.shape
		except AttributeError:
			self.shape = None

	def __getitem__(self, item):
		return self._get_data(item)

	def _get_data(self, item):
		if self._npy is not None:
			if self.dtype == dict:
				return self._npy[item]
			else:
				# noinspection PyUnresolvedReferences
				return self._npy[item].astype(self.dtype)
		return

	def _load_npy(self):
		if self.npy_file is not None:
			_load = pjoin(self.npy_dir, self.npy_file)
			if self.dtype == dict:
				return np.load(_load, allow_pickle=True).item()
			else:
				return np.load(_load, mmap_mode='r')
		return

	def load_ants(self, pixeltype: str = 'float'):
		if self.nii_file is not None:
			_load = pjoin(self.cfg.raw_dir, self.nii_file)
			return ants.image_read(_load, pixeltype=pixeltype)
		return

	def _get_files(self):
		cond = (
			self.space == 'CCF' and
			self.mode in ['ca2', 'bold']
		)
		if cond:
			candidates = fnmatch.filter(
				os.listdir(self.npy_dir),
				f"{self.key}*.npy",
			)
			self.npy_file = self._get_f(candidates)
			self.nii_file = None
		else:
			candidates = fnmatch.filter(
				os.listdir(self.cfg.raw_dir),
				f"{'_'.join(self.key.split('_')[:-1])}*.nii.gz",
			)
			self.nii_file = self._get_f(candidates)
			self.npy_file = None
		return

	def _get_f(self, candidates):
		try:
			return next(
				f for f in candidates if
				self.key == xtract_key(f) and
				all(k in f for k in self._patterns)
			)
		except (StopIteration, FileNotFoundError):
			return None

	def _get_dtype(self):
		if self.space == 'individual':
			if self.mode in ['mask_ca2', 'mask_bold']:
				return bool
			elif self.mode in ['fineatlas2d_ca2', 'fineatlas3d_bold']:
				return int
			elif self.mode in ['referenceimage_ca2', 'T1w', 'ca2', 'bold']:
				return float
			else:
				msg = f"invalid mode encountered: '{self.mode}'"
				raise ValueError(msg)
		elif self.space == 'CCF':
			return float
		else:
			raise NotImplementedError


class FuncData(ImagingData):
	def __init__(
			self,
			key: str,
			mode: str,
			space: str,
			cfg: Config,
			task: str = None,
			desc: str = None,
	):
		_allowed_modes = ['ca2', 'bold']
		assert mode in _allowed_modes,\
			f"allowed modes:\n{_allowed_modes}"

		if task in ['rest', 'led']:
			patterns = [f"task-{task}"]
		else:
			patterns = []

		if desc is not None:
			patterns += [f"_desc-{desc}_"]

		super(FuncData, self).__init__(key, mode, space, cfg, patterns)
		self.task = self._get_task()

	def _get_task(self):
		if self.npy_file is not None:
			file_name = self.npy_file
		else:
			file_name = self.nii_file
		if file_name is not None:
			f = file_name.split('_')
			task = next(
				x for x in f
				if 'task' in x
			)
			return task.split('-')[-1]
		return


class AnatData(ImagingData):
	def __init__(
			self,
			key: str,
			mode: str,
			cfg: Config,
			patterns: List[str] = None,
	):
		_allowed_modes = [
			'fineatlas2d_ca2',
			'fineatlas3d_bold',
			'mask_ca2',
			'mask_bold',
			'atlas2d-inv',  # TODO: fix this
			'atlas3d-inv',
			'referenceimage_ca2',
			'T1w',
		]
		assert mode in _allowed_modes,\
			f"allowed modes:\n{_allowed_modes}"

		if mode in ['atlas2d-inv', 'atlas3d-inv']:
			space = 'CCF'
		else:
			space = 'individual'

		if 'mask' in mode or 'referenceimage' in mode:
			desc, mode = mode.split('_')
			if patterns is None:
				patterns = [f"desc-{desc}"]
			elif isinstance(patterns, list):
				patterns += [f"desc-{desc}"]
			else:
				raise ValueError

		super(AnatData, self).__init__(key, mode, space, cfg, patterns)


def xtract_key(file_name):
	key_ex = filter(
		lambda item: item[0] in ['sub', 'ses', 'run'],
		[e.split('-') for e in file_name.split('_')]
	)
	key_ex = '_'.join(['-'.join(e) for e in key_ex])
	return key_ex
