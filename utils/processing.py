from .generic import *
import nibabel as nib
from pprint import pprint
from sklearn.cluster import KMeans
from scipy import stats as sp_stats
from skimage.measure import block_reduce
from model.configuration import Config
from model.dataset import xtract_key


def save_tsnr():
	pass


# TODO: Sept 2021, clean this up, get rid of the old functions

# TODO: define new process data function
#  -> functional data is processed (ROIs averaged)
#  -> Anat data is not processed, just copy over
#  The allen based atlas files are cllaed ara2d, ara3d

# TODO: Oct 17: this should rename files so the
#  full key (sub, ses, run) is in the beginning of the file
#  See _get_files(self) in dataset.py

# Or just run this at the end of the work:

def rename(cfg):
	for name in sorted(os.listdir(cfg.npy_dir)):
		key_x = xtract_key(name)
		new_name = '_'.join([
			e for e in name.split('_') if
			e not in key_x.split('_')
		])
		new_name = '_'.join([key_x, new_name])
		if name != new_name:
			os.rename(
				src=pjoin(cfg.npy_dir, name),
				dst=pjoin(cfg.npy_dir, new_name),
			)
	return


def nii2npy(
		cfg: Config = None,
		apply_mask: bool = True,
		verbose: bool = True, ):
	cfg = cfg if cfg else Config()

	files = []
	for i in pathlib.Path(cfg.raw_dir).glob('**/*.nii.gz'):
		files.append(i.name)

	if len(files) != len(set(files)) and verbose:
		msg = 'Warning: {:d} duplicate files detected'
		msg = msg.format(len(files) - len(set(files)))
		print(msg)

	files = sorted(set(files))
	part_files = sorted(filter(lambda e: 'part-' in e, files))
	non_part_files = [x for x in files if x not in part_files]
	part_files_unique = set(
		'_'.join(filter(
			lambda e: 'part' not in e,
			item.split('_')
		)) for item in part_files
	)

	# save all single part files
	pbar = tqdm(
		non_part_files,
		total=len(non_part_files),
		dynamic_ncols=True,
		leave=verbose,
		position=0,
	)
	for f in tqdm(non_part_files, desc=''):
		msg = f"processing 1 part files: {f}"
		pbar.set_description(msg)
		try:
			img = nib.load(pjoin(cfg.raw_dir, f))
			# TODO: here get key and bring it to first
			save_obj(
				obj=img.get_fdata().squeeze(),
				file_name='{:s}.npy'.format(f.split('.')[0]),
				save_dir=cfg.npy_dir,
				mode='np',
				verbose=verbose,
			)
		except FileNotFoundError:
			if verbose:
				print(f"Warning! file not found:\n{f}")

	# save 3 part ca files
	pbar = tqdm(
		part_files_unique,
		total=len(part_files_unique),
		dynamic_ncols=True,
		leave=verbose,
		position=0,
	)
	for file in pbar:
		msg = f"processing 3 part ca2 files: {file}"
		pbar.set_description(msg)
		kws = file.split('_')
		ca_file_parts = sorted(
			x for x in part_files if
			all(k in x for k in kws)
		)
		if len(ca_file_parts) != 3 and verbose:
			msg = f"Warning! Ca2 part file missing:\n{file}"
			print(msg)

		ca_data_list = []
		for ca_file in ca_file_parts:
			try:
				with open(pjoin(cfg.raw_dir, ca_file)) as f:
					ca_data_list.append(nib.load(f.name).get_fdata().squeeze())
			except FileNotFoundError:
				if verbose:
					print('Warning! file not found: {:s}'.format(ca_file))
		save_obj(
			obj=np.concatenate(ca_data_list, axis=-1),
			file_name='{:s}.npy'.format(file.split('.')[0]),
			save_dir=cfg.npy_dir,
			mode='np',
			verbose=verbose,
		)

	if apply_mask:
		apply_masks_npy(cfg=cfg, verbose=verbose)

	return


def apply_masks_npy(cfg: Config = None, verbose: bool = True):
	cfg = cfg if cfg else Config()

	for sub in tqdm(cfg.sub_ids, desc='block reduce & apply mask'):
		for ses in cfg.ses_ids:
			kws = [
				'sub-{:s}{:02d}'.format(cfg.group, sub),
				'ses-{:d}'.format(ses),
				'.npy',
			]

			# ca
			mask_file = next(
				f for f in os.listdir(cfg.npy_dir)
				if all(k in f for k in kws + ['mask', 'ca2'])
			)
			anat_file = next(
				f for f in os.listdir(cfg.npy_dir)
				if all(k in f for k in kws + ['referenceimage', 'ca2'])
			)
			mask = np.load(pjoin(cfg.npy_dir, mask_file), mmap_mode='r')
			x = np.load(pjoin(cfg.npy_dir, anat_file), mmap_mode='r+')

			block_size = tuple(i//j for i, j in zip(x.shape, mask.shape))
			if any(i != 1 for i in block_size):
				x = block_reduce(x, block_size=block_size, func=np.mean)
			x *= mask.astype(bool)

			save_obj(
				obj=x.astype(float),
				file_name=anat_file,
				save_dir=cfg.npy_dir,
				mode='np',
				verbose=verbose,
			)

			# bold
			mask_file = next(
				f for f in os.listdir(cfg.npy_dir)
				if all(k in f for k in kws + ['mask', 'bold'])
			)
			anat_file = next(
				f for f in os.listdir(cfg.npy_dir)
				if all(k in f for k in kws + ['T1w'])
			)
			mask = np.load(pjoin(cfg.npy_dir, mask_file), mmap_mode='r')
			x = np.load(pjoin(cfg.npy_dir, anat_file), mmap_mode='r+')

			block_size = tuple(i//j for i, j in zip(x.shape, mask.shape))
			if any(i != 1 for i in block_size):
				x = block_reduce(x, block_size=block_size, func=np.mean)
			x *= mask.astype(bool)

			save_obj(
				obj=x.astype(float),
				file_name=anat_file,
				save_dir=cfg.npy_dir,
				mode='np',
				verbose=verbose,
			)
	return


def process_data_old(config: Config):
	skip = ['mask', 'fineatlas', 'referenceimage']
	shared_names = []
	for i in pathlib.Path(config.raw_dir).glob('**/*.gz'):
		if any(x in i.name for x in skip):
			continue
		meta = i.name.split('_')
		sub = next(filter(lambda x: 'sub' in x, meta))
		ses = next(filter(lambda x: 'ses' in x, meta))
		run = next(filter(lambda x: 'run' in x, meta))
		task = next(filter(lambda x: 'task' in x, meta))
		shared_names.append('_'.join([sub, ses, task, run]))
	shared_names = sorted(list(np.unique(shared_names)))

	# make processed dir if does not exist
	# os.makedirs(config.npy_dir, exist_ok=True)

	# process and save files
	for keyword in tqdm(shared_names):
		selected_files = pathlib.Path(config.raw_dir).glob('**/*{:s}*.nii.gz'.format(keyword))
		selected_files = list(map(lambda x: str(x), selected_files))
		try:
			bold_load_file = next(filter(lambda x: 'bold' in x, selected_files))
		except StopIteration:
			continue

		meta = bold_load_file.split('/')[-1].split('_')
		sub = next(filter(lambda x: 'sub' in x, meta))
		ses = next(filter(lambda x: 'ses' in x, meta))
		run = next(filter(lambda x: 'run' in x, meta))
		task = next(filter(lambda x: 'task' in x, meta))
		key = '_'.join([sub, ses, run, task])

		# bold
		try:
			save_obj(
				obj=nib.load(bold_load_file).get_fdata(),
				file_name=key + '_bold.npy',
				save_dir=config.npy_dir,
				mode='np',
				verbose=True,
			)
		except FileNotFoundError:
			print('Warning! file not found:\n{:s}'.format(bold_load_file))
			continue

		# ca
		ca_load_files = sorted([item for item in selected_files if 'ca' in item])
		ca_data_list = []
		for ca_file in ca_load_files:
			try:
				with open(ca_file) as f:
					ca_data_list.append(nib.load(f.name).get_fdata())
			except FileNotFoundError:
				print('Warning! file not found: {:s}'.format(ca_file))

		save_obj(
			obj=np.concatenate(ca_data_list, axis=-1),
			file_name=key + '_ca.npy',
			save_dir=config.npy_dir,
			mode='np',
			verbose=True,
		)


def _organize(
		config: Config,
		subject: int,
		session: int,
		run: int,
		lowcut: float,
		highcut: float,
		bandpass_order: int,
		bandpass: bool,
		normalize: bool,
		verbose: bool = False, ):

	data = load_ca_with_extras(
		config=config,
		subject=subject,
		session=session,
		run=run,
		filter_data=True,
		compute_kmeans=config.atlas_mode == 'kmeans',
		verbose=verbose,
	)
	if data is None:
		return None

	if config.atlas_mode == 'allen':
		_atlas = 'atlas'
	elif config.atlas_mode == 'kmeans':
		_atlas = 'atlas_kmeans'
	else:
		msg = 'invalid atlas mode encountered: {:s}'
		raise RuntimeError(msg.format(config.atlas_mode))

	region_ids = np.unique(data[_atlas])
	region_ids = [item for item in region_ids if item > 0]

	ca_organized = {}
	for key in region_ids:
		condition = data[_atlas] == key
		if condition.sum():
			x = data['ca'][condition].mean(0)
		else:
			continue

		if bandpass:
			x = apply_bandpass(
				x=x,
				fs=config.ca_fs,
				lowcut=lowcut,
				highcut=highcut,
				order=bandpass_order,
			)

		if normalize:
			x = sp_stats.zscore(x)

		if np.isnan(x).sum() > 0:
			msg = 'Warning! nan found.\n'
			msg += "sub: {:d}, ses: {:d}, run: {:d}, region '{:d}'"
			msg = msg.format(subject, session, run, key)
			print(msg)
		else:
			ca_organized[key] = x

	output = {'ca': ca_organized}
	output.update({k: v for k, v in data.items() if k != 'ca'})
	return output


def organize_ca_data(
		config: Config = None,
		lowcut: float = 0.002,
		highcut: float = 1.0,
		bandpass_order: int = 2,
		bandpass: bool = True,
		normalize: bool = True,
		verbose: bool = False, ):

	config = Config() if config is None else config

	for subject in tqdm(range(1, config.num_subjects+1)):
		for session in tqdm(range(1, config.num_sessions+1), leave=False):
			for run in range(1, config.num_runs+1):

				# load & organize
				data_organized = _organize(
					config=config,
					subject=subject,
					session=session,
					run=run,
					lowcut=lowcut,
					highcut=highcut,
					bandpass_order=bandpass_order,
					bandpass=bandpass,
					normalize=normalize,
					verbose=verbose,
				)
				if data_organized is None:
					continue

				# TODO: current normalize: local and temporal
				#  what else can be done:
				#  1) global (the whole run has mean zero and std = 1
				#  2) spatial: each npix_x * npix_y frame has mean
				#  zero std = 1 ( does not make sense )
				# TODO: add normalize_global and local, possibly inside config

				# save results
				save_dir = 'atlas-'
				if config.atlas_mode == 'kmeans':
					save_dir += '{:d}'.format(config.num_regions)
				else:
					save_dir += 'allen'
				if bandpass:
					msg = '_filtered_low-{:.3f}_high-{:.3f}_ord-{:d}'
					msg = msg.format(lowcut, highcut, bandpass_order)
					save_dir += msg
				else:
					save_dir += '_unfiltered'
				if normalize:
					save_dir += '_normalized'
				else:
					save_dir += '_unnormalized'
				save_dir = pjoin(config.processed_dir, save_dir)
				os.makedirs(save_dir, exist_ok=True)

				file_name = data_organized['file_name'].split('.')[0] + '_organized.npy'
				save_obj(
					obj=data_organized,
					file_name=file_name,
					save_dir=save_dir,
					mode='np',
					verbose=verbose,
				)
	return


def load_organized_data(
		folder: str,
		config: Config = None,
		subject: int = 1,
		session: int = 1,
		run: int = 1,
		modality: str = 'ca',
		verbose: bool = False, ):

	config = Config() if config is None else config
	load_dir = pjoin(config.processed_dir, folder)
	keywords = [
		'sub-SLC{:02d}'.format(subject),
		'ses-{:d}'.format(session),
		'run-{:d}'.format(run),
		'organized',
		str(modality),
	]

	try:
		file_name = next(x for x in os.listdir(load_dir) if all(k in x for k in keywords))
	except StopIteration:
		if verbose:
			print('Warning! File not found, moving on . . .\n{}\n'.format(keywords))
		return None

	load_file = pjoin(load_dir, file_name)
	return np.load(load_file, allow_pickle=True).item()


def load_ca_with_extras(
		config: Config = None,
		subject: int = 1,
		session: int = 1,
		run: int = 1,
		filter_data: bool = True,
		compute_kmeans: bool = True,
		random_state: int = 42,
		verbose: bool = True, ):

	config = Config() if config is None else config
	keywords = [
		'sub-SLC{:02d}'.format(subject),
		'ses-{:d}'.format(session),
		'run-{:d}'.format(run),
		'ca',
	]

	# ca
	try:
		ca_file = next(x for x in os.listdir(config.npy_dir) if all(k in x for k in keywords))
	except StopIteration:
		print('Warning! File not found, moving on . . .\n{}\n'.format(keywords))
		return None
	good_timepoints = (config.run_duration - config.exclude) * config.ca_fs

	ca = np.load(pjoin(config.npy_dir, ca_file), mmap_mode='r')
	ca = ca[..., 0, range(ca.shape[-1] - good_timepoints, ca.shape[-1])]

	# remove run from keywords
	keywords = [item for item in keywords if 'run' not in item]

	# atlas
	atlas = None
	keywords_atlas = keywords + ['fineatlas2d']
	for i in pathlib.Path(config.raw_dir).glob('**/*.gz'):
		if all(k in i.name for k in keywords_atlas):
			atlas_file = str(i)
			atlas = nib.load(atlas_file).get_fdata().squeeze()
	if atlas is None:
		print('Warning! atlas file not found, keywords:\n{}'.format(keywords_atlas))

	# reference image
	ref = None
	keywords_ref = keywords + ['referenceimage']
	for i in pathlib.Path(config.raw_dir).glob('**/*.gz'):
		if all(k in i.name for k in keywords_ref):
			ref_file = str(i)
			ref = nib.load(ref_file).get_fdata().squeeze()

	if ref is None:
		msg = 'Warning! referenceimage file not found, keywords:\n{}'
		print(msg.format(keywords_ref))
	else:
		ref /= ref.std()

	# mask
	mask = None
	keywords_mask = keywords + ['mask']
	for i in pathlib.Path(config.raw_dir).glob('**/*.gz'):
		if all(k in i.name for k in keywords_mask):
			mask_file = str(i)
			mask = nib.load(mask_file).get_fdata().squeeze()
	if mask is None:
		msg = 'Warning! referenceimage file not found, keywords:\n{}'
		print(msg.format(keywords_mask))
	else:
		mask[mask != 0.] = 1.  # binarize mask

	# process
	if mask is not None:
		mask = np.logical_and(mask, np.abs(ca).mean(-1) > 0)
		atlas *= mask
		if ref is not None:
			block_size = (ref.shape[0]//mask.shape[0], ref.shape[1]//mask.shape[1])
			ref = block_reduce(ref, block_size=block_size, func=np.mean)
			ref *= mask

	# TODO: is there an inconsistency between mask and atlas?

	# filter boundaries
	if filter_data:
		ca, atlas, ref, mask = filter_boundaries(
			x_list=[ca, atlas, ref, mask],
			mask=mask,
			dim=2,
		)

	# compute kmeans parcellation
	if compute_kmeans:
		atlas_kmeans, kmeans_centers = kmeans_parcellation(
			mask=mask,
			num_regions=config.num_regions,
			random_state=random_state,
		)
	else:
		atlas_kmeans = []
		kmeans_centers = []

	if verbose:
		print("Loading:\n{:s}".format(ca_file))
		msg = '\n-->  {:d} s  <-- excluded from the beginning of each run. Combined runs duration:'
		msg = msg.format(config.exclude)
		print(msg)
		convert_time(ca.shape[-1] / config.ca_fs)

	output = {
		'ca': np.array(ca, dtype=float),
		'atlas': np.array(atlas, dtype=int),
		'ref': np.array(ref, dtype=float),
		'mask': np.array(mask, dtype=bool),
		'atlas_kmeans': np.array(atlas_kmeans, dtype=int),
		'kmeans_centers': np.array(kmeans_centers, dtype=float),
		'file_name': ca_file,
	}
	return output


def load_with_extras(
		config: Config = None,
		subject: int = 1,
		session: int = 1,
		run: int = 1,
		filter_data: bool = True,
		compute_kmeans: bool = True,
		random_state: int = 42,
		verbose: bool = True, ):

	config = Config() if config is None else config
	ca = _load('ca', config, subject, session, run, filter_data, compute_kmeans, random_state, verbose)
	bold = _load('bold', config, subject, session, run, filter_data, compute_kmeans, random_state, verbose)
	return ca, bold


def _load(
		mode: str,
		config: Config(),
		subject: int,
		session: int,
		run: int,
		filter_data: bool = True,
		compute_kmeans: bool = True,
		random_state: int = 42,
		verbose: bool = True, ):

	assert mode in ['ca', 'bold'], "Invalid mode encountered: {}".format(mode)
	keywords = [
		'sub-SLC{:02d}'.format(subject),
		'ses-{:d}'.format(session),
		'run-{:d}'.format(run),
		mode,
	]
	try:
		file = next(x for x in os.listdir(config.npy_dir) if all(k in x for k in keywords))
	except StopIteration:
		print('Warning! File not found, moving on . . .\n{}\n'.format(keywords))
		return None

	fs = config.ca_fs if mode == 'ca' else config.bold_fs
	good_timepoints = (config.run_duration - config.exclude) * fs

	x = np.load(pjoin(config.npy_dir, file), mmap_mode='r')
	x = x[..., range(x.shape[-1] - good_timepoints, x.shape[-1])]
	x = x.squeeze()

	print(mode, x.shape)

	# remove run from keywords
	keywords = [item for item in keywords if 'run' not in item]

	# atlas
	atlas = []
	keywords_atlas = keywords + ['fineatlas2d' if mode == 'ca' else 'fineatlas3d']
	for i in pathlib.Path(config.raw_dir).glob('**/*.gz'):
		if all(k in i.name for k in keywords_atlas):
			atlas_file = str(i)
			atlas = nib.load(atlas_file).get_fdata().squeeze()
	if not len(atlas):
		print('Warning! atlas file not found, keywords:\n{}'.format(keywords_atlas))

	# reference image
	ref = []
	keywords_ref = keywords + ['referenceimage']
	for i in pathlib.Path(config.raw_dir).glob('**/*.gz'):
		if all(k in i.name for k in keywords_ref):
			ref_file = str(i)
			ref = nib.load(ref_file).get_fdata().squeeze()
	if not len(ref):
		print('Warning! referenceimage file not found, keywords:\n{}'.format(keywords_ref))
	else:
		ref /= ref.std()

	# mask
	mask = []
	keywords_mask = keywords + ['mask']
	for i in pathlib.Path(config.raw_dir).glob('**/*.gz'):
		if all(k in i.name for k in keywords_mask):
			mask_file = str(i)
			mask = nib.load(mask_file).get_fdata().squeeze()
	if not len(mask):
		print('Warning! referenceimage file not found, keywords:\n{}'.format(keywords_mask))
	else:
		mask[mask != 0.] = 1.  # binarize mask

	# process
	if len(mask):
		mask = np.logical_and(mask, np.abs(x).mean(-1) > 0)
		atlas *= mask
		if len(ref):
			# block_size = (ref.shape[0]//mask.shape[0], ref.shape[1]//mask.shape[1])
			block_size = tuple(i//j for i, j in zip(ref.shape, mask.shape))
			print(block_size)
			ref = block_reduce(ref, block_size=block_size, func=np.mean)
			ref *= mask

	# TODO: is there an inconsistency between mask and atlas?

	# filter boundaries
	if filter_data and len(mask):
		x, atlas, ref, mask = filter_boundaries(
			x_list=[x, atlas, ref, mask],
			mask=mask,
		)

	# compute kmeans parcellation
	if compute_kmeans:
		atlas_kmeans, kmeans_centers = kmeans_parcellation(
			mask=mask,
			num_regions=config.num_regions,
			random_state=random_state,
		)
	else:
		atlas_kmeans = []
		kmeans_centers = []

	if verbose:
		print("Loading:\n{:s}".format(file))
		msg = '\n-->  {:d} s  <-- excluded from the beginning of each run. Combined runs duration:'
		msg = msg.format(config.exclude)
		print(msg)
		convert_time(x.shape[-1] / fs)

	output = {
		'x': np.array(x, dtype=float),
		'atlas': np.array(atlas, dtype=int),
		'ref': np.array(ref, dtype=float),
		'mask': np.array(mask, dtype=bool),
		'atlas_kmeans': np.array(atlas_kmeans, dtype=int),
		'kmeans_centers': np.array(kmeans_centers, dtype=float),
		'file_name': file,
	}
	return output


def kmeans_parcellation(
		mask: np.ndarray,
		num_regions: int = 1000,
		random_state: int = 42, ):

	nonzero_pix = list(zip(*np.where(mask)))
	nonzero_pix = np.reshape(nonzero_pix, (-1, 2))

	kmeans = KMeans(
		n_clusters=num_regions,
		random_state=random_state,
	).fit(nonzero_pix)
	roi = kmeans.predict(nonzero_pix)

	roi_unique = []
	for lbl in roi:
		if lbl not in roi_unique:
			roi_unique.append(lbl)

	roi_relabeld = np.zeros_like(roi)
	for new_lbl, lbl in enumerate(roi_unique):
		roi_relabeld[roi == lbl] = new_lbl + 1

	atlas_kmeans_dict = collections.defaultdict(list)
	for pixel, lbl in zip(nonzero_pix, roi_relabeld):
		atlas_kmeans_dict[lbl].append(tuple(pixel))
	atlas_kmeans_dict = dict(atlas_kmeans_dict)

	atlas_kmeans = np.zeros(mask.shape, dtype=int)
	for region_id, pixel_list in atlas_kmeans_dict.items():
		atlas_kmeans[tuple(zip(*pixel_list))] = region_id

	# TODO: fix the cluster centers, it needs reordering (like roi_relabeled)
	return atlas_kmeans, kmeans.cluster_centers_


def load_ca_data(
		config: Config = None,
		keywords: list = None,
		verbose=True, ):

	config = Config() if config is None else config
	keywords = ['ca', 'ses-1', 'sub-SLC06', 'rest'] if keywords is None else keywords
	if 'ca' not in keywords:
		keywords.append('ca')

	def _filter_fn(x):
		return all(k in x for k in keywords)
	files = sorted(list(filter(_filter_fn, os.listdir(config.npy_dir))))

	good_timepoints = (config.run_duration - config.exclude) * config.ca_fs

	data_list = []
	for load_filename in files:
		ca = np.load(pjoin(config.npy_dir, load_filename), mmap_mode='r')
		data_list.append(ca[..., 0, range(ca.shape[-1] - good_timepoints, ca.shape[-1])])
	ca = np.concatenate(data_list, axis=-1)
	ca = filter_boundaries([ca], mask=np.abs(ca).mean(-1))[0]

	if verbose:
		print("Data from {:d} runs loaded:\n".format(len(files)))
		pprint(files)

		msg = '\n-->  {:d} s  <-- excluded from the beginning of each run. Combined runs duration:'
		msg = msg.format(config.exclude)
		print(msg)
		convert_time(ca.shape[-1] / config.ca_fs)

	return ca


def downsample_data(x: np.ndarray, size: int = 32):
	if len(x.shape) == 3:
		npix_x, npix_y, nt = x.shape
		widths = [int(np.ceil(item / size)) for item in [npix_x, npix_y]]

		downsampled = np.zeros((size, size, nt))
		for i in range(size):
			for j in range(size):
				data = x[widths[0] * i: widths[0] * (i + 1), widths[1] * j: widths[1] * (j + 1), :]
				num_nonzero_mean = sum(abs(data).mean(-1).flatten() != 0)

				if num_nonzero_mean > 0:
					downsampled[i, j, :] = data.reshape(-1, nt).sum(0) / num_nonzero_mean
				else:
					downsampled[i, j, :] = 0.

	elif len(x.shape) == 4:
		raise NotImplementedError

	else:
		raise ValueError("Only 2 and 3 spatial dims supported.")

	return downsampled
