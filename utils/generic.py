import re
import os
import ants
import nrrd
import json
import h5py
import pickle
import joblib
import shutil
import random
import pathlib
import inspect
import logging
import argparse
import warnings
import operator
import functools
import itertools
import collections
import numpy as np
import pandas as pd
import bottleneck as bn
from rich import print
from scipy import fft as sp_fft
from scipy import signal as sp_sig
from scipy import stats as sp_stats
from scipy import ndimage as sp_img
from scipy import optimize as sp_optim
from scipy.spatial import distance as sp_dist
from statsmodels.stats.multitest import multipletests
from skimage.transform import rescale as sk_rescale
from skimage.transform import resize as sk_resize
from sklearn.preprocessing import Normalizer
from numpy.ma import masked_where as mwh
from os.path import join as pjoin
from datetime import datetime
from tqdm import tqdm
from typing import *
import numbers


def flatten_arr(
		x: np.ndarray,
		ndim_end: int = 1,
		ndim_start: int = 0, ):
	shape = x.shape
	assert 0 <= ndim_end <= len(shape)
	assert 0 <= ndim_start <= len(shape)
	if ndim_end + ndim_start >= len(shape):
		return x

	shape_flat = shape[:ndim_start] + (-1,)
	for i, d in enumerate(shape):
		if i >= len(shape) - ndim_end:
			shape_flat += (d,)
	return x.reshape(shape_flat)


def avg(
		x: np.ndarray,
		ndim_end: int = 2,
		ndim_start: int = 0,
		fn: Callable = bn.nanmean, ) -> np.ndarray:
	dims = range(ndim_start, x.ndim - ndim_end)
	dims = sorted(dims, reverse=True)
	for axis in dims:
		x = fn(x, axis=axis)
	return x


def get_tval(
		dof: int = 9,
		ci: float = 0.95,
		two_sided: bool = True, ):
	if two_sided:
		ci = (1 + ci) / 2
	return sp_stats.t.ppf(ci, dof)


def contig_segments(mask: np.ndarray):
	censored = np.where(mask == 0)[0]
	looper = itertools.groupby(
		enumerate(censored),
		lambda t: t[0] - t[1],
	)
	segments = []
	for k, g in looper:
		s = map(operator.itemgetter(1), g)
		segments.append(list(s))
	return segments


def unique_idxs(
		obj: np.ndarray,
		filter_zero: bool = True, ):
	idxs = pd.DataFrame(obj.flat)
	idxs = idxs.groupby([0]).indices
	if filter_zero:
		idxs.pop(0, None)
	return idxs


def all_equal(iterable):
	g = itertools.groupby(iterable)
	return next(g, True) and not next(g, False)


def np_nans(shape: Union[int, Iterable[int]]):
	if isinstance(shape, np.ndarray):
		shape = shape.shape
	arr = np.empty(shape, dtype=float)
	arr[:] = np.nan
	return arr


def make_logger(
		name: str,
		path: str,
		module: str,
		verbose: bool = True, ) -> logging.Logger:
	os.makedirs(path, exist_ok=True)
	logger = logging.getLogger(module)
	logger.setLevel(logging.INFO)
	name = f"{name}_({now()}).log"
	file = pjoin(path, name)
	file_handler = logging.FileHandler(file)
	if verbose:
		msg = f"logger '{name}' created at\n{path}"
		print(msg)

	formatter = logging.Formatter(
		'%(asctime)s : %(levelname)s : %(name)s : %(message)s')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


def make_pdf(
		dist: Union[str, Callable],
		params: Tuple[float],
		whis: Tuple[float, float] = (0.001, 0.999),
		size: int = 10000, ):
	arg = params[:-2]
	loc = params[-2]
	scale = params[-1]
	if isinstance(dist, str):
		dist = getattr(sp_stats, dist)
	if arg:
		rv = dist(*arg, loc=loc, scale=scale)
	else:
		rv = dist(loc=loc, scale=scale)
	start = rv.ppf(whis[0])
	end = rv.ppf(whis[1])
	x = np.linspace(start, end, size)
	return rv, pd.Series(rv.pdf(x), x)


def get_rng(
		x: Union[int, np.random.Generator, random.Random] = 42,
		use_np: bool = True, ):
	if isinstance(x, int):
		if use_np:
			return np.random.default_rng(seed=x)
		else:
			return random.Random(x)
	elif isinstance(x, (np.random.Generator, random.Random)):
		return x
	else:
		print('Warning, invalid random state. returning default')
		return np.random.default_rng(seed=42)


def setup_kwargs(defaults, kwargs):
	if not kwargs:
		return defaults
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	return kwargs


def get_default_params(f: Callable):
	params = inspect.signature(f).parameters
	return {
		k: p.default for
		k, p in params.items()
	}


def filter_kwargs(fn, kw: dict = None):
	if not kw:
		return {}
	try:
		params = inspect.signature(fn).parameters
		return {
			k: v for k, v
			in kw.items()
			if k in params
		}
	except ValueError:
		return kw


def filter_boundaries(
		x_list: Union[np.ndarray, List[np.ndarray]],
		mask: np.ndarray = None, ):
	if not isinstance(x_list, list):
		x_list = [x_list]
	if mask is None:
		# noinspection PyTypeChecker
		mask = functools.reduce(
			np.logical_or,
			[x.astype(bool) for x in x_list]
		)
	dim = len(mask.shape)

	if dim == 2:
		bad_x_indices = np.where(mask.mean(1) == 0)[0]
		bad_y_indices = np.where(mask.mean(0) == 0)[0]
		bad_z_indices = []
	elif dim == 3:
		bad_x_indices = np.where(mask.mean(-1).mean(-1) == 0)[0]
		bad_y_indices = np.where(mask.mean(0).mean(-1) == 0)[0]
		bad_z_indices = np.where(mask.mean(0).mean(0) == 0)[0]
	else:
		raise ValueError("wrong dim: {}".format(dim))

	filtered = []
	for x in x_list:
		if dim == 2:
			x_filtered = np.delete(np.delete(
				x, bad_y_indices, axis=1), bad_x_indices, axis=0)
		elif dim == 3:
			x_filtered = np.delete(np.delete(np.delete(
				arr=x, obj=bad_z_indices, axis=2,
			), obj=bad_y_indices, axis=1,
			), obj=bad_x_indices, axis=0,
			)
		else:
			raise ValueError("wrong dim: {}".format(dim))
		filtered.append(x_filtered)
	return filtered if len(filtered) > 1 else filtered.pop()


# TODO: improve this fn to do all the other stuff
def train_test_split(n_samples: int, xv_folds: int = 5, which_fold: int = 0):
	which_fold %= xv_folds
	num_tst_indxs = int(np.ceil(n_samples / xv_folds))

	start = which_fold * num_tst_indxs
	stop = min((which_fold + 1) * num_tst_indxs, n_samples)
	tst_indxs = range(start, stop)
	trn_indxs = np.delete(range(n_samples), tst_indxs)

	assert not set(tst_indxs).intersection(set(trn_indxs))
	return np.array(tst_indxs), np.array(trn_indxs)


# TODO: by other stuff this is what i mean
"""
selected_keys = [k for k in within_all if 'run' in k]

led_ids = [i for i, k in enumerate(selected_keys) if 'led' in k]
rest_ids = [i for i, k in enumerate(selected_keys) if 'rest' in k]
led_ids, rest_ids = np.array(led_ids), np.array(rest_ids)

len(led_ids), len(rest_ids)

xv_folds = 10
which_fold = 0

led_tst, led_trn = train_test_split(len(led_ids), xv_folds=xv_folds, which_fold=which_fold)
rest_tst, rest_trn = train_test_split(len(rest_ids), xv_folds=xv_folds, which_fold=which_fold)

led_tst, led_trn = led_ids[led_tst], led_ids[led_trn]
rest_tst, rest_trn = rest_ids[rest_tst], rest_ids[rest_trn]

tst = np.concatenate([led_tst, rest_tst])
trn = np.concatenate([led_trn, rest_trn])

assert not set(tst).intersection(set(trn))

"""


def smoothen(arr: np.ndarray, filter_sz: int = 5):
	shape = arr.shape
	assert 1 <= len(shape) <= 2, "1 <= dim <= 2d"

	kernel = np.ones(filter_sz) / filter_sz
	if len(shape) == 1:
		return np.convolve(arr, kernel, mode='same')
	else:
		smoothed = np.zeros(arr.shape)
		for i in range(arr.shape[0]):
			smoothed[i] = np.convolve(arr[i], kernel, mode='same')
		return smoothed


def downsample(data, xy, xbins, ybins, normal=True):
	xbins = sorted(xbins)
	ybins = sorted(ybins)
	assert len(xbins) == len(ybins)
	nbins = len(xbins)

	delta_x = (max(xbins) - min(xbins)) / nbins
	delta_y = (max(ybins) - min(ybins)) / nbins

	nc = xy.shape[0]
	downsampled = np.zeros((nbins, nbins))
	_norm = np.zeros((nbins, nbins))
	for cell in range(nc):
		x, y = xy[cell]

		if x == max(xbins):
			bin_i = -1
		else:
			bin_i = int(np.floor(
				(x - min(xbins)) / delta_x
			))
		if y == max(ybins):
			bin_j = -1
		else:
			bin_j = int(np.floor(
				(y - min(ybins)) / delta_y
			))

		downsampled[bin_j, bin_i] += data[cell]
		_norm[bin_j, bin_i] += 1

	if normal:
		return np.divide(downsampled, np.maximum(_norm, 1e-8))
	else:
		return downsampled


def reset_df(df: pd.DataFrame, downcast: str = 'integer') -> pd.DataFrame:
	df.reset_index(drop=True, inplace=True)
	if downcast != 'none':
		df = df.apply(pd.to_numeric, downcast=downcast, errors="ignore")
	return df


# noinspection PyTypeChecker
def save_obj(
		obj: Any,
		file_name: str,
		save_dir: str,
		mode: str = None,
		verbose: bool = True, ):
	_allowed_modes = [
		'npy', 'df',
		'pkl', 'joblib',
		'html', 'json', 'txt',
	]
	_ext = file_name.split('.')[-1]
	if _ext in _allowed_modes:
		mode = _ext
	else:
		if mode is None:
			msg = 'invalid file extension: '
			msg += f"{_ext}, mode: {mode}"
			raise RuntimeError(msg)
		else:
			file_name = f"{file_name}.{mode}"
	assert mode in _allowed_modes,\
		f"available modes:\n{_allowed_modes}"

	path = pjoin(save_dir, file_name)
	op_mode = 'w' if mode in ['html', 'json', 'txt'] else 'wb'
	with open(path, op_mode) as f:
		if mode == 'npy':
			np.save(f.name, obj)
		elif mode == 'df':
			pd.to_pickle(obj, f)
		elif mode == 'pkl':
			pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
		elif mode == 'joblib':
			joblib.dump(obj, f)
		elif mode == 'html':
			f.write(obj)
		elif mode == 'json':
			json.dump(obj, f, indent=4)
		elif mode == 'txt':
			for line in obj:
				f.write(line)
		else:
			msg = 'invalid mode, available options:'
			msg += f"\n{_allowed_modes}"
			raise RuntimeError(msg)
	if verbose:
		print(f"[PROGRESS] '{file_name}' saved at\n{save_dir}")
	return path


def merge_dicts(dict_list: List[dict], verbose: bool = False) -> Dict[str, list]:
	merged = collections.defaultdict(list)
	dict_items = map(operator.methodcaller('items'), dict_list)
	iterable = itertools.chain.from_iterable(dict_items)
	for k, v in tqdm(iterable, disable=not verbose, leave=False, desc="...merging dicts"):
		merged[k].extend(v)
	return dict(merged)


def rm_dirs(base_dir: str, dirs: List[str], verbose: bool = True):
	for x in dirs:
		dirpath = pathlib.Path(base_dir, x)
		if dirpath.exists() and dirpath.is_dir():
			shutil.rmtree(dirpath)
	if verbose:
		print("[PROGRESS] removed {} folders at {:s}".format(dirs, base_dir))


def _butter_bandpass(lowcut, highcut, fs, order=2):
	nyq = 0.5 * fs
	lowcut = lowcut / nyq
	highcut = highcut / nyq
	output = sp_sig.butter(
		N=order,
		Wn=[lowcut, highcut],
		btype='bandpass',
		output='ba',
		fs=fs,
	)
	# noinspection PyTupleAssignmentBalance
	b, a = output
	return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
	b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
	zi = sp_sig.lfilter_zi(b, a)
	y, zo = sp_sig.lfilter(b, a, data, zi=zi * data[0])
	return y, zo


def _apply_bandpass(
		x: np.ndarray,
		fs: int,
		lowcut: float = 0.01,
		highcut: float = 4.0,
		order: int = 2, ):
	orig_shape = x.shape
	x = x.reshape(-1, orig_shape[-1])  # time is last dimension
	x_bp = np.zeros(x.shape)
	for i in range(x.shape[0]):
		x_bp[i], _ = _butter_bandpass_filter(
			data=x[i],
			lowcut=lowcut,
			highcut=highcut,
			fs=fs,
			order=order,
		)
	return x_bp.reshape(orig_shape)


def now(include_hour_min: bool = False):
	s = "%Y_%m_%d"
	if include_hour_min:
		s += ",%H:%M"
	return datetime.now().strftime(s)


def convert_time(time_in_secs: float, verbose: bool = True):
	d = time_in_secs // 86400
	h = (time_in_secs - d * 86400) // 3600
	m = (time_in_secs - d * 86400 - h * 3600) // 60
	s = time_in_secs - d * 86400 - h * 3600 - m * 60

	d, h, m, s = tuple(map(lambda x: int(x), [d, h, m, s]))
	msg = "{:02d} - {:02d}:{:02d}:{:02d}  (dd - hh:mm:ss)\n"
	msg = msg.format(d, h, m, s)
	if verbose:
		print(msg)
	else:
		return msg


def isfloat(string: str):
	try:
		float(string)
		return True
	except ValueError:
		return False
