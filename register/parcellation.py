from analysis.helper import *
from model.configuration import Config
from register.register import ants, reset_ants_img
from register.atlas import Allen, get_subtree, mask_left_right
from scipy import spatial as sp_spat
from sklearn.cluster import KMeans
import sklearnex

_ATTRS = [
	'cols',
	'brain',
	'cortex',
	'subcortex',
	'cortex-top',
	'brain-eroded',
	'nonzero-idxs_parcel2d-flat',
	'nonzero-idxs_parcel2d-top',
	'nonzero-idxs_parcel3d',
	'region-idxs_parcel2d-flat',
	'region-idxs_parcel2d-top',
	'region-idxs_parcel3d',
	'shape_parcel2d-flat',
	'shape_parcel2d-top',
	'shape_parcel3d',
]


class Parcellation(object):
	def __init__(
			self,
			cfg: Config = None,
			allen: Allen = None,
			full_load: bool = False,
			verbose: bool = True,
			show: bool = False,
	):
		super(Parcellation, self).__init__()

		self.cfg = cfg if cfg else Config()
		self.al = allen if allen else Allen(
			resolution=cfg.resolution,
			full_load=full_load,
		)
		self.verbose = verbose

		self.df = None
		self.results = {}
		self.fit_meta = {}
		self.results2d = {}  # TODO: get rid of this
		self.load_parcel(full_load)
		if show:
			_ = self.show()
			_ = self.show2d()

	def __call__(self, item):
		return self.get(item)

	def __getitem__(self, item):
		return self.results.get(item)

	def _fit_columnar(self):
		if self.verbose:
			print('[PROGRESS] fit columnar: initiated')

		al10 = None

		f = 'parcel2d-flat_10um.nii.gz'
		f = pjoin(self.cfg.parcel_dir, f)
		if os.path.isfile(f):
			parcel2d = ants.image_read(
				filename=f, pixeltype='unsigned int')
		else:
			al10 = al10 if al10 else Allen(
				resolution=10,
				proj_mode='dorsal_flatmap',
				full_load=True,
			)
			output = kmeans_parcellation(
				mask=al10.lookup > 0,
				n_layers=self.cfg.ll,
				n_hemisphere_regions=self.cfg.nn,
				random_state=self.cfg.random_state,
				lr_axis=1,
			)
			parcel2d = output['parcel']
			if self.verbose:
				print('[PROGRESS] kmeans fit: done')

			parcel2d = reset_ants_img(parcel2d, 10)
			ants.image_write(parcel2d, f)

			f = 'flatmap_10um.nii.gz'
			f = pjoin(self.cfg.base_dir, f)
			if not os.path.isfile(f):
				flatmap = al10.cortical_map(
					volume=al10.template,
					proj_mode='dorsal_flatmap',
					agg_func=np.nanmax,
				)
				flatmap = reset_ants_img(flatmap, 10)
				ants.image_write(flatmap, f)

			f = 'topview_10um.nii.gz'
			f = pjoin(self.cfg.base_dir, f)
			if not os.path.isfile(f):
				topview = al10.cortical_map(
					volume=al10.template,
					proj_mode='top_view',
					agg_func=np.nanmax,
				)
				topview = reset_ants_img(topview, 10)
				ants.image_write(topview, f)

			if self.verbose:
				print('[PROGRESS] save 2D, 10um: done')

		parcel2d = parcel2d.numpy()

		f = 'parcel3d_10um.nii.gz'
		f = pjoin(self.cfg.parcel_dir, f)
		if os.path.isfile(f):
			parcel3d = ants.image_read(
				filename=f, pixeltype='unsigned int')
		else:
			al10 = al10 if al10 else Allen(
				resolution=10,
				proj_mode='dorsal_flatmap',
				full_load=True,
			)
			regions = parcel2d[parcel2d.nonzero()]
			regions = np.unique(regions)
			parcel3d = np.zeros_like(al10.template)
			for i in regions:
				idxs = np.where(parcel2d == i)
				for item in al10.lookup[idxs]:
					path = al10.paths[item]
					path = path[path.nonzero()]
					parcel3d.flat[path] = i
			if self.verbose:
				print('[PROGRESS] 2d --> 3d: done')

			parcel3d = reset_ants_img(parcel3d, 10)
			ants.image_write(parcel3d, f)

			f = 'template_10um.nii.gz'
			f = pjoin(self.cfg.base_dir, f)
			if not os.path.isfile(f):
				template = reset_ants_img(al10.template, 10)
				ants.image_write(template, f)

			if self.verbose:
				print('[PROGRESS] save 3D, 10um: done')

		# generate top view
		f = 'parcel2d-top_10um.nii.gz'
		f = pjoin(self.cfg.parcel_dir, f)
		if not os.path.isfile(f):
			al10 = al10 if al10 else Allen(
				resolution=10,
				proj_mode='top_view',
				full_load=False,
			)
			top2d = al10.cortical_map(
				volume=parcel3d.numpy(),
				proj_mode='top_view',
				agg_func=np.max,
			)
			top2d = fill_gaps(top2d, None, False)
			top2d = reset_ants_img(top2d, 10)
			ants.image_write(top2d, f)

		# downsample
		if self.al.template is None:
			self.al.load_everything()
		tgt = reset_ants_img(
			img=self.al.template,
			spacing=self.al.mcc.resolution,
		)
		parcel3d_down = ants.resample_image_to_target(
			image=parcel3d, target=tgt,
			interp_type='genericLabel',
		)
		parcel3d_down = parcel3d_down.numpy()
		if self.verbose:
			print('[PROGRESS] downsample: done')

		# Fill gaps cols
		columns = parcel3d_down.astype('uint32')
		columns = fill_gaps(
			x=columns,
			valid=self.al.get_masks('Isocortex'),
			finalize=True,
		)

		# Add layers
		cortex = self._add_layers(columns)
		regions = cortex[cortex.nonzero()]
		regions = np.unique(regions)

		expected = range(1, self.num_ctx_rois() + 1)
		missing = [
			i for i in expected
			if i not in regions
		]
		self.fit_meta['ctx_missing'] = missing
		if missing:
			msg = 'Warning! missing regions:'
			msg += f"\n'{missing}'"
			print(msg)
			missing = self.complement(missing)
			cortex[np.isin(cortex, missing)] = 0

		regions = cortex[cortex.nonzero()]
		regions = np.unique(regions)
		top, top_regions = self.top_only(cortex)

		self.results = {
			'cols': columns,
			'cortex': cortex,
			'cortex-top': top,
			**self._get_ravel_idxs(),
		}
		if self.verbose:
			self.check_symmetry(regions)
			print('[PROGRESS] fit columnar: done')
		return

	def _fit_subcortical(
			self,
			frac: float = 0.4,
			nb_std: float = 3, ):
		if self.verbose:
			print('[PROGRESS] fit subcortical: initiated')
		regions = self['cortex'][self['cortex'].nonzero()]
		avg_sz = np.mean([
			(self['cortex'] == r).sum()
			for r in np.unique(regions)
		])
		major = self.al.get_major_12()
		if 'Isocortex' in major:
			major.pop(major.index('Isocortex'))
		masks = {
			acro: self.al.get_masks(acro)
			for acro in major
		}
		n_rois = {
			acro: int(np.round(
				frac * m.sum() / avg_sz / 2
			)) for acro, m in masks.items()
		}
		self.fit_meta['subctx_num'] = n_rois
		# fit subcortex
		subctx_bad = {}
		subctx_regions = {}
		subctx = np.zeros_like(self['cortex'])
		count = self.num_ctx_rois()
		for acro, n in n_rois.items():
			if acro == 'Isocortex':
				continue
			results = kmeans_parcellation(
				mask=masks[acro],
				n_hemisphere_regions=n,
				symmetrize=True,
				lr_axis=2,
			)
			assert 2 * n == len(results['regions'])
			sizes = {
				r: (results['parcel'] == r).sum()
				for r in results['regions']
			}
			sz = list(sizes.values())
			mu, sd = np.mean(sz), np.std(sz)
			bad_l = [
				r for r, s in sizes.items() if (
					s <= mu - nb_std * sd and
					r in results['regions'][:n]
				)
			]
			bad_r = [
				r for r, s in sizes.items() if (
					s <= mu - nb_std * sd and
					r in results['regions'][n:]
				)
			]
			bad = sorted(set(
				bad_l + [r + n for r in bad_l]).union(
				bad_r + [r - n for r in bad_r]
			))
			subctx_bad[acro] = len(bad)
			regions = [
				r for r in results['regions']
				if r not in bad
			]
			mapping = {
				i + count: r for i, r in
				enumerate(regions, start=1)
			}
			for i, r in mapping.items():
				subctx[results['parcel'] == r] = i
			subctx_regions[acro] = list(mapping)
			count += len(regions)
		self.fit_meta['subctx_bad'] = subctx_bad
		self.results['subcortex'] = subctx
		if self.verbose:
			print('[PROGRESS] fit subcortical: done')
		self.df = self._create_df(subctx_regions)
		if self.verbose:
			print('[PROGRESS] create regions df: done')
		return

	def _fit_spatial(self):
		output = kmeans_parcellation(
			mask=self.al.get_masks('Isocortex'),
			n_hemisphere_regions=self.cfg.nn,
			random_state=self.cfg.random_state,
			lr_axis=2,
		)
		self.results = {
			'cortex': output['parcel'].astype('uint32'),
			'cortex-eroded': output['eroded'].astype('uint32'),
			'cortex_centers': output['cluster_centers'],
			'regions-cortex': output['regions'],
		}
		return {}

	def fit_parcel(
			self,
			radius: str = 'infer',
			force_fit: bool = False,
			**kwargs, ):
		if not force_fit:
			loaded = all(self.load_parcel(False))
			if loaded:
				if self.verbose:
					print('Parcellation loaded succesfully')
			else:
				force_fit = True
				if self.verbose:
					print('Parcellation not found, initiating fit')

		self.fit_meta['time_start'] = now(True)
		if force_fit:
			if self.cfg.parcel_mode == 'columnar':
				self._fit_columnar()
			elif self.cfg.parcel_mode == 'spatial':
				self._fit_spatial()
			else:
				raise NotImplementedError
			self._fit_subcortical(**kwargs)
			brain = np.zeros_like(self['cortex'])
			for k in ['cortex', 'subcortex']:
				idxs = self[k].nonzero()
				brain[idxs] = self[k][idxs]
			self.results['brain'] = brain
			eroded, radii = self.erode(radius)
			if self.verbose:
				print('[PROGRESS] erode: done')
			self.results['brain-eroded'] = eroded
			self.fit_meta['eroded-radii'] = radii
			self.fit_meta['time_end'] = now(True)
			self.save()
		return

	# noinspection PyTypeChecker
	def load_parcel(self, full: bool):
		if full:
			essentials = _ATTRS
		else:
			exclude = [
				'cols',
				'cortex',
				'cortex-top',
				'subcortex',
			]
			essentials = [
				e for e in _ATTRS
				if e not in exclude
			]
		loaded = []
		for item in essentials:
			f = f"{item}_{self.cfg.resolution}um.npy"
			f = pjoin(self.cfg.parcel_dir, f)
			if os.path.isfile(f):
				loaded.append(True)
				try:
					self.results[item] = np.load(
						f, mmap_mode='r')
				except ValueError:
					d = np.load(
						f, allow_pickle=True)
					d = dict(d.item())
					self.results[item] = d
			else:
				loaded.append(False)
		try:
			self.df = pd.read_pickle(
				pjoin(self.cfg.parcel_dir, 'regions.df'))
			loaded.append(True)
		except FileNotFoundError:
			loaded.append(False)
		return loaded

	def save(self):
		for item in _ATTRS:
			save_obj(
				obj=self.results.get(item),
				file_name=f"{item}_{self.cfg.resolution}um",
				save_dir=self.cfg.parcel_dir,
				mode='npy',
				verbose=self.verbose,
			)
		save_obj(
			obj=self.df,
			file_name='regions',
			save_dir=self.cfg.parcel_dir,
			mode='df',
			verbose=self.verbose,
		)
		save_obj(
			obj=self.fit_meta,
			file_name='fit_meta',
			save_dir=self.cfg.parcel_dir,
			mode='npy',
			verbose=self.verbose,
		)
		if self.verbose:
			print('[PROGRESS] save results: done.')
		return

	def load_top_flat(self):
		f = 'parcel2d-top_10um.nii.gz'
		f = pjoin(self.cfg.parcel_dir, f)
		parcel_top = ants.image_read(
			f, pixeltype='unsigned int')
		parcel_top = parcel_top.numpy()

		f = 'parcel2d-flat_10um.nii.gz'
		f = pjoin(self.cfg.parcel_dir, f)
		parcel_flat = ants.image_read(
			f, pixeltype='unsigned int')
		parcel_flat = parcel_flat.numpy()
		return parcel_top, parcel_flat

	def decode(self, regions, return_arr_only=True):
		# TODO: is incorrect in the latest version
		#  DO NOT USE.
		ll = 3 if self.cfg.ll == 0 else self.cfg.ll

		def _decode(i):
			lr = (i - 1) // (self.cfg.nn * ll)
			i -= self.cfg.nn * ll * lr
			lay = (i - 1) // self.cfg.nn
			col = i - lay * self.cfg.nn
			return lr + 1, col, lay
		decode_ = np.vectorize(_decode)

		if isinstance(regions, (int, np.uint32, np.int64)):
			regions = [regions]

		if not len(regions):
			return []

		hcl = decode_(regions)
		hcl = np.array(list(zip(*hcl)))
		if return_arr_only:
			return hcl
		else:
			hcl_d = dict(zip(regions, hcl))
			return hcl, hcl_d

	def encode(self, *args) -> Union[int, np.ndarray]:
		if len(args) == 1:
			x = args[0]
			if isinstance(x, str):
				try:
					hh, cc, ll = tuple(map(
						lambda s: int(s),
						x.split('-'),
					))
				except ValueError:
					return int(x.split('-')[-1])
			else:
				hh = x[:, 0]
				cc = x[:, 1]
				ll = x[:, 2]
		elif len(args) == 3:
			hh = np.array(args[0])
			cc = np.array(args[1])
			ll = np.array(args[2])
		else:
			raise RuntimeError('invalid input')

		if self.cfg.ll == 0:
			return self.cfg.nn * (ll + 3 * (hh - 1)) + cc
		else:
			return self.cfg.nn * (ll + self.cfg.ll * (hh - 1)) + cc

	def get(self, r: Union[int, Iterable[int]]):
		if isinstance(r, (int, numbers.Integral)):
			assert 1 <= r <= len(self.df)
			return dict(self.df.loc[r - 1])
		else:
			return self.df.loc[self.df['id'].isin(r)]

	def get_id(self, lr: int, cc: int, ll: int):
		assert lr in self.df['hemis'].unique()
		assert cc in self.df['region'].unique()
		assert ll in self.df['layer'].unique()
		region_id = self.df.loc[
			(self.df['hemis'] == lr) &
			(self.df['region'] == cc) &
			(self.df['layer'] == ll),
			'id',
		].item()
		return region_id

	def rank_order_regions(self, cortex, subcortex=None):
		# TODO: no longer accurate after latest version
		#  DO NOT USE
		def _sort_fn(x):
			lr, cc, ll = x
			return int(lr * 1e6) + int(cc * 1e3) + ll

		hcl = self.decode(cortex)
		hcl = sorted(hcl, key=_sort_fn)
		hcl = [e.reshape(1, -1) for e in hcl]
		hcl = np.concatenate(hcl)
		l_ids = np.where(hcl[:, 0] == 1)[0]
		r_ids = np.where(hcl[:, 0] == 2)[0]

		regions_l = [
			f"{lr}-{cc}-{ll}" for
			lr, cc, ll in hcl[l_ids]
		]
		regions_r = [
			f"{lr}-{cc}-{ll}" for
			lr, cc, ll in hcl[r_ids]
		]
		if subcortex:
			regions_l += [f"1-{ii}" for ii in subcortex]
			regions_r += [f"2-{ii}" for ii in subcortex]
		regions_ranked = regions_l + regions_r
		return regions_ranked

	def check_symmetry(self, regions):
		if isinstance(regions, dict):
			regions = list(regions)
		hcl = self.decode(regions)
		left = hcl[hcl[:, 0] == 1]
		right = hcl[hcl[:, 0] == 2]
		l_ids = self.encode(left)
		r_ids = self.encode(right)
		try:
			delta = collections.Counter(r_ids - l_ids)
		except ValueError:
			delta = 'different num left and right...'
		msg = 'num regions,\t'
		msg += f"left: {len(left)},  right: {len(right)}\n"
		msg += f"region id diff,\t\tr - l:  {delta}\n"
		print(msg)
		return

	def complement(self, regions):
		if not isinstance(regions, list):
			regions_list = list(regions)
		else:
			regions_list = regions.copy()
		comp = []
		for hh, cc, ll in self.decode(regions_list):
			mirror = self.encode(hh % 2 + 1, cc, ll)
			comp.append(mirror)
		regions_list += comp
		regions_list = sorted(set(regions_list))
		return np.array(regions_list, dtype='uint32')

	def top_only(self, obj: np.ndarray):
		regions = np.unique(obj[obj.nonzero()])
		hcl = self.decode(regions)
		deep_regions = regions[hcl[:, -1] > 0]
		x = obj.copy()
		x[np.isin(x, deep_regions)] = 0
		return x, np.unique(x[x.nonzero()])

	def num_ctx_rois(self):
		num = self.cfg.nn
		num *= self.cfg.ll * 2
		return num

	def _add_layers(self, cortex):
		layers = self.al.make_layer_masks()
		regions = cortex[cortex.nonzero()]
		regions = np.unique(regions)
		cortex_layered = np.zeros_like(cortex)
		if self.cfg.ll == 3:
			depths = [
				['1', '2/3', '4'],
				['5'],
				['6a', '6b'],
			]
		elif self.cfg.ll == 2:
			depths = [
				['1', '2/3', '4', '5'],
				['6a', '6b'],
			]
		else:
			raise NotImplementedError
		depths = [functools.reduce(
			np.logical_or,
			[m for k, m in layers.items() if k in e]
		) for e in depths]

		for r in regions:
			lr, cc, _ = self.decode(r).ravel()
			_m = cortex == r
			for ll, combo_m in enumerate(depths):
				m = np.logical_and(_m, combo_m)
				r_new = self.encode(lr, cc, ll)
				cortex_layered[m] = r_new
		return cortex_layered

	def _get_ravel_idxs(self):
		obj_names = [
			'parcel2d-top',
			'parcel2d-flat',
			'parcel3d',
		]
		output = {}
		for name in obj_names:
			obj = f"{name}_10um.nii.gz"
			obj = pjoin(self.cfg.parcel_dir, obj)
			obj = ants.image_read(obj, pixeltype='unsigned int')
			obj = obj.numpy()
			nonzero = obj.nonzero()
			nonzero = np.ravel_multi_index(nonzero, obj.shape)
			idxs = unique_idxs(obj.flat[nonzero])
			output[f"region-idxs_{name}"] = idxs
			output[f"nonzero-idxs_{name}"] = nonzero
			output[f"shape_{name}"] = obj.shape
		# noinspection PyTypeChecker
		return dict(sorted(output.items()))

	def _get_cutoff(self, cutoff, hcl, sizes_np):
		if isinstance(cutoff, int):
			cutoff *= (200 / self.cfg.resolution) ** 3
		elif isinstance(cutoff, (str, float)):
			last_layer = self.cfg.ll - 1
			last_layer = hcl[:, -1] == last_layer
			if isinstance(cutoff, float):
				q = cutoff % 1.0
				cutoff = np.quantile(sizes_np[last_layer], q)
			elif cutoff == 'infer':
				cutoff = np.quantile(sizes_np[last_layer], 0.5)
			else:
				raise RuntimeError(f"invalid cutoff: '{cutoff}'")
		else:
			raise RuntimeError(f"invalid cutoff: '{cutoff}'")
		return int(cutoff)

	def erode(self, radius: Union[str, float] = 'infer'):
		radii = {}
		eroded = np.zeros_like(self['brain'])
		for acro in self.df['acro'].unique():
			df = self.df.loc[self.df['acro'] == acro]
			if acro == 'Isocortex':
				for layer in range(self.cfg.ll):
					ids = df.loc[
						df['layer'] == layer,
						'id',
					].values
					x = self['cortex'].copy()
					m = np.isin(x, ids)
					x[~m] = 0
					x_er, r = erode_parcellation(x, radius)
					radii[f"{acro}-{layer}"] = r
					eroded[m] = x_er[m]
			else:
				ids = df['id'].values
				x = self['subcortex'].copy()
				m = np.isin(x, ids)
				x[~m] = 0
				x_er, r = erode_parcellation(x, radius)
				radii[acro] = r
				eroded[m] = x_er[m]
		return eroded.astype('uint32'), radii

	def _create_df(self, subctx_regions):
		_, hcl_dict = self.decode(
			range(1, self.num_ctx_rois() + 1),
			return_arr_only=False,
		)
		df = []
		for r, (lr, cc, ll) in hcl_dict.items():
			df.append({
				'id': [r],
				'acro': ['Isocortex'],
				'hemis': [lr],
				'region': [cc],
				'layer': [ll],
			})
		for acro, region_ids in subctx_regions.items():
			num = len(region_ids)
			n = num // 2
			df.append({
				'id': region_ids,
				'acro': [acro] * num,
				'hemis': [1] * n + [2] * n,
				'region': region_ids[:n] + region_ids[:n],
				'layer': [-1] * num,
			})
		return pd.DataFrame(merge_dicts(df))

	def prepare_major_masks(self):  # TODO: no longer needed
		major_masks = get_major_masks(self.al)
		major_colors = self.al.get(major_masks)
		major_colors = dict(zip(
			major_colors['acronym'],
			major_colors['rgb_triplet'],
		))
		major_colors = {
			acro: rgb2hex([e / 255 for e in c])
			for acro, c in major_colors.items()
		}
		major_colors = {
			acro: major_colors.get(acro) for
			acro in major_masks
		}
		cmap = make_cmap(
			ramp_colors=list(major_colors.values()),
			n_colors=len(major_colors),
			name='major_cmap',
			show=False,
		)
		output = {
			'major_masks': major_masks,
			'major_colors': major_colors,
			'cmap': cmap,
		}
		return output

	def fill_rois_2d(self, dry_run: bool = False):
		paths = r'*indiv2d_cortex.nii.gz'
		paths = pathlib.Path(self.cfg.masks_dir).rglob(paths)
		for p in sorted(paths):
			img = ants.image_read(
				str(p), pixeltype='unsigned int')
			obj = img.numpy()
			regions = np.unique(obj[obj.nonzero()])

			hcl = self.decode(regions)
			lay0 = regions[hcl[:, -1] == 0]
			lay1 = regions[hcl[:, -1] == 1]
			if not len(lay1):
				continue
			hcl1 = self.decode(lay1)
			hcl1[:, -1] = 0
			lay1_to_0 = self.encode(hcl1)
			miss0 = [e for e in lay1_to_0 if e not in lay0]

			if miss0:
				flat = 'parcel2d-flat_10um.nii.gz'
				flat = pjoin(self.cfg.parcel_dir, flat)
				flat = ants.image_read(
					flat, pixeltype='unsigned int').numpy()
				top = 'parcel2d-top_10um.nii.gz'
				top = pjoin(self.cfg.parcel_dir, top)
				top = ants.image_read(
					top, pixeltype='unsigned int').numpy()

				fig, axes = create_figure(
					nrows=1,
					ncols=4,
					figsize=(13, 2.8),
					width_ratios=[1, 2] * 2,
				)

				data2plt = np.ma.masked_where(
					~np.isin(top, regions), top)
				axes[0].imshow(top > 0, cmap='Greys_r')
				axes[0].imshow(data2plt, cmap='prism')
				axes[0].set_ylabel('top (before)')

				data2plt = np.ma.masked_where(
					~np.isin(flat, regions), flat)
				axes[1].imshow(flat > 0, cmap='Greys_r')
				axes[1].imshow(data2plt, cmap='prism')
				axes[1].set_title('flat (before)')

				hcl_miss = self.decode(miss0)
				hcl_miss[:, -1] = 1
				miss1 = self.encode(hcl_miss)
				for i, j in zip(miss0, miss1):
					obj[obj == j] = i
				regions = np.unique(obj[obj.nonzero()])

				data2plt = np.ma.masked_where(
					~np.isin(top, regions), top)
				axes[2].imshow(top > 0, cmap='Greys_r')
				axes[2].imshow(data2plt, cmap='prism')
				axes[2].set_ylabel('top (after)')

				data2plt = np.ma.masked_where(
					~np.isin(flat, regions), flat)
				axes[3].imshow(flat > 0, cmap='Greys_r')
				axes[3].imshow(data2plt, cmap='prism')
				axes[3].set_title('flat (after)')

				fig.suptitle(f"{p.name}.  missing: {miss0}")
				plt.show()

				if not dry_run:
					img = reset_ants_img(
						img=obj,
						spacing=img.spacing,
						origin=img.origin,
						use_um=False,
					)
					ants.image_write(img, str(p))
		return

	def show_accepted_regions_2d(self, mice, thres: float = 0.1):
		flat = pjoin(self.cfg.parcel_dir, 'parcel2d-flat_10um.nii.gz')
		flat = ants.image_read(flat, pixeltype='unsigned int').numpy()
		top = pjoin(self.cfg.parcel_dir, 'parcel2d-top_10um.nii.gz')
		top = ants.image_read(top, pixeltype='unsigned int').numpy()
		nrows = len(mice.ref)
		fig, axes = create_figure(
			nrows=nrows,
			ncols=4,
			figsize=(13, 2.6 * nrows),
			width_ratios=[1, 2, 1, 1],
		)
		dlist = []
		regions_all = {}
		for ii, key in enumerate(mice.ref):
			roi_2d = f"{key}_space-indiv2d_cols.nii.gz"
			roi_2d = pjoin(mice.cfg.masks_dir, roi_2d)
			roi_2d = ants.image_read(roi_2d, pixeltype='unsigned int')
			roi_2d = reset_ants_img(
				img=roi_2d.numpy().squeeze(),
				spacing=roi_2d.spacing[:2],
				origin=0, use_um=False,
			)
			ref = mice.ref[key].load_ants('float')
			ref = reset_ants_img(
				img=ref.numpy().squeeze(),
				spacing=ref.spacing[:2],
				origin=0, use_um=False,
			)
			mask = mice.mask2d[key].load_ants('unsigned int')
			roi_2d = ants.resample_image_to_target(
				image=roi_2d, target=mask,
				interp_type='genericLabel',
			)
			ref = ants.resample_image_to_target(
				image=ref, target=mask,
				interp_type='linear',
			)
			roi_2d *= mask
			ref *= mask

			x = roi_2d.numpy().astype(int).squeeze()
			sizes = collections.Counter(x[x.nonzero()])
			dlist.append({
				'key': [key] * len(sizes),
				'region': sizes.keys(),
				'pixels': sizes.values(),
			})

			regions2d = sorted(sizes)
			regions_all[key] = regions2d

			ref_tr, roi_tr, mask_tr = filter_boundaries(
				[ref.numpy(), roi_2d.numpy(), mask.numpy()], mask.numpy())
			ref_tr = ref_tr.astype(float)
			roi_tr = roi_tr.astype('uint32')
			mask_tr = mask_tr.astype(bool)

			if self.verbose:
				axes[ii, 0].set_ylabel(f"{key}\n", fontsize=15)

				data2plt = np.ma.masked_where(
					~np.isin(top, regions2d), top)
				axes[ii, 0].imshow(top > 0, cmap='Greys_r')
				axes[ii, 0].imshow(data2plt, cmap='prism')

				data2plt = np.ma.masked_where(
					~np.isin(flat, regions2d), flat)
				axes[ii, 1].imshow(flat > 0, cmap='Greys_r')
				axes[ii, 1].imshow(data2plt, cmap='prism')

				axes[ii, 2].imshow(
					np.ma.masked_where(~mask_tr, ref_tr), cmap='Greys_r')
				axes[ii, 2].imshow(
					np.ma.masked_where(roi_tr.astype(bool), roi_tr),
					alpha=0.6, cmap='prism')

				axes[ii, 3].imshow(
					np.ma.masked_where(~mask_tr, ref_tr), cmap='Greys_r')
				axes[ii, 3].imshow(
					np.ma.masked_where(~roi_tr.astype(bool), roi_tr),
					alpha=0.6, cmap='prism')
		df_sz = pd.DataFrame(merge_dicts(dlist))

		if self.verbose:
			plt.show()

		regions_agg = []
		for item in regions_all.values():
			regions_agg += item
		freq = collections.Counter(regions_agg)
		freq = {
			i: v / len(regions_all)
			for i, v in freq.items()
		}
		freq = {
			i: freq.get(i, 0.0) for i in
			self.complement(list(freq))
		}

		freq_top = np.zeros(top.shape)
		freq_flat = np.zeros(flat.shape)
		for i, val in freq.items():
			freq_top[top == i] = val
			freq_flat[flat == i] = val

		thres_values = [0.2, 0.4, 0.5, 0.8, 0.9]
		thres_values = [t for t in thres_values if t != thres]
		thres_values = [thres] + thres_values
		nrows = 1 + len(thres_values)
		fig, axes = create_figure(
			nrows=nrows,
			ncols=2,
			figsize=(10, 2.7 * nrows),
			width_ratios=[1, 2],
		)

		if self.verbose:
			axes[0, 0].set_ylabel('frequency [%]\n', fontsize=15)

			data2plt = np.ma.masked_where(
				~top.astype(bool), freq_top * 100)
			im = axes[0, 0].imshow(data2plt, cmap='nipy_spectral')
			plt.colorbar(im, ax=axes[0, 0])

			data2plt = np.ma.masked_where(
				~flat.astype(bool), freq_flat * 100)
			im = axes[0, 1].imshow(data2plt, cmap='nipy_spectral')
			plt.colorbar(im, ax=axes[0, 1])

			for i, th in enumerate(thres_values, start=1):
				axes[i, 0].set_ylabel(f"threshold = {th}\n", fontsize=15)

				x = freq_top.copy()
				x[x > th] = 1.
				x[x <= th] = 0.
				data2plt = np.ma.masked_where(~top.astype(bool), x)
				im = axes[i, 0].imshow(data2plt, cmap='nipy_spectral')
				plt.colorbar(im, ax=axes[i, 0])

				x = freq_flat.copy()
				x[x > th] = 1.
				x[x <= th] = 0.
				data2plt = np.ma.masked_where(~flat.astype(bool), x)
				im = axes[i, 1].imshow(data2plt, cmap='nipy_spectral')
				plt.colorbar(im, ax=axes[i, 1])
			plt.show()

		rej = dict(filter(
			lambda t: t[1] < thres,
			freq.items()
		))
		rej = self.complement(sorted(rej))
		acc = [
			i for i in freq
			if i not in rej
		]

		if self.verbose:
			msg = '\n\n\n\n'
			msg += f"accept ratio: {100 * len(acc) / len(freq):0.0f} %, "
			msg += f"reject ratio: {100 * len(rej) / len(freq):0.0f} %, "
			msg += f"using a presence threshold of {thres * 100} %\n"
			msg += f"a total of {len(acc)} regions were accepted\n"
			print(msg)

			fig, axes = create_figure(
				nrows=1,
				ncols=4,
				figsize=(13, 2.3),
				width_ratios=[1, 2] * 2,
			)
			data2plt = np.ma.masked_where(
				~np.isin(top, acc), top)
			axes[0].imshow(top > 0, cmap='Greys_r')
			axes[0].imshow(data2plt, cmap='prism')
			data2plt = np.ma.masked_where(
				~np.isin(flat, acc), flat)
			axes[1].imshow(flat > 0, cmap='Greys_r')
			axes[1].imshow(data2plt, cmap='prism')
			data2plt = np.ma.masked_where(
				~np.isin(top, rej), top)
			axes[2].imshow(top > 0, cmap='Greys_r')
			axes[2].imshow(data2plt, cmap='prism')
			data2plt = np.ma.masked_where(
				~np.isin(flat, rej), flat)
			axes[3].imshow(flat > 0, cmap='Greys_r')
			axes[3].imshow(data2plt, cmap='prism')
			plt.show()

		output = {
			'freq': freq,
			'df_sz': df_sz,
			'rejected': rej,
			'accepted': acc,
		}
		return output

	def trim_boundaries_2d(self, x: Union[List, Dict], mask: np.ndarray):
		if mask is None:
			mask = self.results2d.get('atlas').astype(bool)
		if isinstance(x, dict):
			x_list = list(x.values())
			x_trm = filter_boundaries(x_list, mask)
			x_trm = dict(zip(x, x_trm))
		elif isinstance(x, list):
			x_list = [np.array(e) for e in x]
			x_trm = filter_boundaries(x_list, mask)
		elif isinstance(x, np.ndarray):
			x_trm = filter_boundaries(x, mask)
		else:
			msg = f"Wrong data type: '{type(x).__name__}'"
			raise RuntimeError(msg)
		return x_trm

	def get_region_overlap(self, thresh: float = 20.):
		major_masks = self.prepare_major_masks()
		major_masks = major_masks['major_masks']
		major_masks = self.al.get_masks(
			query=major_masks,
			as_objectmap=False,
		)
		region_overlap = {}
		cortex_regions = self['cortex'][self['cortex'].nonzero()]
		cortex_regions = np.unique(cortex_regions)
		cortex_regions = {
			i: self['cortex'] == i
			for i in cortex_regions
		}
		pbar = tqdm(cortex_regions.items())
		for region, m in pbar:
			pbar.set_description(f"region: {region}")
			_s = {}
			for k, M in major_masks.items():
				_s[k] = overlap_score(m, M) * 100
			_s = sorted(_s.items(), key=lambda x: x[1], reverse=True)
			region_overlap[region] = {
				k: np.round(s)
				for k, s in _s
				if s > thresh
			}
		return region_overlap

	def show(self, figsize=(10, 4), bins=80):
		accepted, rejected = [], []
		all_distances_accepted = []
		for i, ctr in self.results['cortex_centers'].items():
			num_voxels = np.sum(self.results['cortex'] == i)
			num_accepted = np.sum(self.results['cortex-eroded'] == i)

			accepted.append(num_accepted)
			rejected.append(num_voxels - num_accepted)

			points = np.array(tuple(zip(*np.where(self.results['cortex-eroded'] == i))))
			d = sp_dist.cdist(ctr.reshape(-1, points.shape[1]), points).flatten()
			all_distances_accepted.extend(list(d))

		fig, axes = create_figure(1, 2, figsize, style='whitegrid')

		sns.histplot(rejected, color='C7', label='rejected', alpha=0.3, ax=axes[0])
		sns.histplot(accepted, color='C2', label='accepted', alpha=0.7, ax=axes[0])
		_min, _median, _max = np.min(accepted), np.median(accepted), np.max(accepted)
		axes[0].axvline(_min, color='C4', ls='--', lw=1.3, label=f"min: {_min}")
		axes[0].axvline(_median, color='C3', ls='--', lw=1.5, label=f"median: {_median}")
		axes[0].axvline(_max, color='C4', ls='--', lw=1.3, label=f"max: {_max}")
		axes[0].set_title('distribution of ROI sizes', fontsize=15, y=1.03)
		axes[0].set_xlabel('num voxels', fontsize=14)
		axes[0].legend()

		y, bin_edges = np.histogram(all_distances_accepted, bins=bins)
		x = (bin_edges[:-1] + bin_edges[1:]) / 2
		m = np.polyfit(x[y > 0], np.log2(y[y > 0]), 1)[0]

		lbl = f"fitted slope (log2): {np.round(m, 2)}"
		sns.histplot(
			all_distances_accepted, color='C6', stat='count',
			kde=True, line_kws={'lw': 4, 'label': lbl}, ax=axes[1]
		)
		axes[1].axvline(
			self.results['radius'], color='k',
			label=f"max radius: {np.round(self.results['radius'], 2)}",
			ls='--', lw=2
		)
		axes[1].set_xscale('log', base=2)
		axes[1].set_yscale('log', base=2)
		axes[1].set_title('voxel distances from ROI ctrs (eroded)', fontsize=15, y=1.03)
		axes[1].set_xlabel('r', fontsize=14)
		axes[1].legend(fontsize=13)

		fig.tight_layout()
		plt.show()
		return fig, axes

	def show2d(self, include: List[str] = None, cmap=None, plt_kwrgs=None, **kwargs):
		kwarg_defaults = {
			'figsize': (11, 5.5),
			'tight_layout': True,
			'constrained_layout': False,
			'width_ratios': [1, 0.1, 1.5, 1.5],
			'fontsize_big': 20,
			'fontsize': 12,
			'y_big': 1.08,
			'y': 1.04,
		}
		for k, v in kwarg_defaults.items():
			if k not in kwargs:
				kwargs[k] = v
		if len(kwargs['width_ratios']) != 4:
			kwargs['width_ratios'] = kwarg_defaults['width_ratios']
		plt_kwrgs = plt_kwrgs if plt_kwrgs else {}

		fig, axes = plt.subplots(
			ncols=4,
			nrows=3,
			figsize=kwargs['figsize'],
			width_ratios=kwargs['width_ratios'],
			tight_layout=kwargs['tight_layout'],
			constrained_layout=kwargs['constrained_layout'],
		)
		gs = axes[0, -1].get_gridspec()
		for ax in axes[:, 1:].flatten():
			ax.remove()
		axbig = fig.add_subplot(gs[:, -2:])
		axbig.set_title(
			label='Annotation 2D',
			fontsize=kwargs['fontsize_big'],
			y=kwargs['y_big'],
		)
		fig, axbig = self.get_annot_fig(fig, axbig, include, cmap, **plt_kwrgs)

		for i in range(3):
			if i == 0:
				x = self.results2d['cortex']
				title = 'total: {:d}'
			elif i == 1:
				x = self.results2d['cortex-eroded']
				title = 'accepted: {:d}'
			else:
				x = self.results2d['cortex_rej']
				title = 'removed: {:d}'
			title = title.format(len(np.unique(x)) - 1)

			mask = self.results2d['cortex'].astype(bool)
			_x = filter_boundaries(x, mask=mask)
			data2plt = np.ma.masked_where(_x == 0, _x)
			axes[i, 0].imshow(
				data2plt, cmap='turbo',
				vmin=1, vmax=self.results['n_cortical_regions'],
			)
			axes[i, 0].set_title(
				label=title,
				fontweight='bold' if i == 1 else 'normal',
				fontsize=kwargs['fontsize'],
				y=kwargs['y'],
			)
			remove_ticks(axes[i, 0])
		plt.show()
		return fig, (axes, axbig)

	def get_annot_fig(self, fig=None, ax=None, include=None, cmap=None, **plt_kwrgs):
		if include is None:
			include = self.results2d['acro2idx']
		else:
			include = [x for x in self.results2d['acro2idx'] if x in include]
		cmap = cmap if cmap else self.results2d['annot_cmap']

		cond = [
			self.results2d['atlas'] == self.results2d['acro2idx'][acro]
			for acro in include
		]
		# noinspection PyTypeChecker
		cond = functools.reduce(np.logical_or, cond)
		cond = np.logical_or(~cond, self.results2d['contour'])
		_atlas2d, _cond = filter_boundaries(
			[self.results2d['atlas'], cond],
			mask=self.results2d['atlas'].astype(bool)
		)
		data2plt = np.ma.masked_where(_cond, _atlas2d)

		cticks = [x for x in self.results2d['acro2idx'].values()]
		ctick_labels = [
			f" {name} ({acro})" if acro in include else ''
			for acro, name in
			zip(self.results2d['acro2idx'], self.results2d['name2idx'])
		]
		fig, ax = plot_annot_2d(
			x=data2plt,
			cticks=cticks,
			ctick_labels=ctick_labels,
			fig=fig,
			ax=ax,
			cmap=cmap,
			**plt_kwrgs,
		)
		return fig, ax


def get_sub_regions(
		cfg: Config,
		allen: Allen,
		remove: List[int] = None,
		return_key: str = 'id', ) -> List[str]:
	remove = remove if remove else [
		1039, 1044, 1120, 1139, 143, 177, 189, 207,
		333, 379, 706, 912, 939, 1044, 1120, 1139, 143,
		177, 207, 333, 379, 531, 706, 912, 939, 982, 934,
	]
	f = pjoin(cfg.base_dir, 'Yale', 'mouseatlas_2020.json')
	with open(f, 'r') as file:
		atlas_xilin = json.load(file)

	iso_subtree, _ = get_subtree(
		allen.bg.hierarchy,
		base_id=allen.get('Isocortex')['id'],
		n_levels=2,
		show=False,
	)
	fine_labels = atlas_xilin['labels']['data'][2]['labels']
	fine_acros = [
		s.split('(')[-1].split(',')[0]
		for s in fine_labels.values()
	]
	acros = list(set(fine_acros))
	structs = allen.s_tree.get_structures_by_acronym(acros)
	subcortical = [
		s[return_key] for s in structs
		if not (s['id'] in iso_subtree or s['id'] in remove)
	]
	return subcortical


def get_cortex_substructs(
		allen: Allen,
		extra_structs: List[str] = None,
		sort: bool = True, ):

	structs = ['Isocortex']
	structs += extra_structs if extra_structs else ['MO', 'SS', 'SSp']
	structs = allen.s_tree.get_structures_by_acronym(structs)

	selected_ids = collections.defaultdict(list)
	for struct in structs:
		subtree, base_lvl = allen.get_subtree(
			base_id=struct['id'], show=False)
		for identifier, node in subtree.nodes.items():
			lvl = allen.bg.hierarchy.level(node.identifier)
			if lvl != base_lvl + 1:
				continue
			selected_ids[struct['acronym']].append(node.identifier)

	sub_structs = allen.s_tree.get_structures_by_id(selected_ids['Isocortex'])
	sub_structs = {
		s['acronym']: allen.mcc.get_structure_mask(s['id'])[0]
		for s in sub_structs
	}
	if sort:
		def _sort_fn(tup):
			mask = tup[1]
			points = np.array(tuple(zip(*np.where(mask))))
			com = points.mean(0)
			return com[0]
		sub_structs = dict(sorted(
			sub_structs.items(),
			key=_sort_fn,
			reverse=True,
		))

	# add extras
	sub_structs = list(sub_structs)
	for acro, ids in selected_ids.items():
		if acro == 'Isocortex':
			continue
		extra_structs = allen.s_tree.get_structures_by_id(ids)
		extra = {
			item['acronym']: allen.mcc.get_structure_mask(item['id'])[0]
			for item in extra_structs
		}
		pos = sub_structs.index(acro) + 1
		sub_structs = sub_structs[:pos] + list(extra) + sub_structs[pos:]
	return sub_structs


def get_major_masks(
		allen: Allen,
		subcortical: List[str] = None,
		sort_cortex: bool = True, ):
	sub = subcortical if subcortical else [
		'PAL',
		'STR',
		'CTXsp',
		'OLF',
		'HIP',
		'RHP',
		'IB',
		'MB',
		'HB',
		'CB',
	]
	return sub + get_cortex_substructs(allen, sort=sort_cortex)


def kmeans_parcellation(
		mask: np.ndarray,
		n_hemisphere_regions: int,
		n_layers: int = 1,
		lr_axis: int = 2,
		symmetrize: bool = True,
		random_state: int = 42, ):

	output_lr = mask_left_right(mask, lr_axis)
	slices_l = output_lr['slices_l']
	slices_r = output_lr['slices_r']
	mask_l = output_lr['mask_l']
	mask_r = output_lr['mask_r']

	# optimizes time
	sklearnex.patch_sklearn(verbose=False)

	if symmetrize:
		mask_r_full = mask_r.copy()
		mask_l_full = mask_l.copy()
		mask_r_full[slices_l] = np.flip(mask_r[slices_r], axis=lr_axis)
		mask_l_full[slices_r] = np.flip(mask_l[slices_l], axis=lr_axis)
		mask_sym = np.logical_and(mask_r_full, mask_l_full)

		mask_l = mask_sym.copy().astype(bool)
		mask_r = mask_sym.copy().astype(bool)
		mask_l[slices_r] = 0
		mask_r[slices_l] = 0

		# do left
		parcel, cluster_centers = _do_kmeans(
			x=mask_l,
			num_regions=n_hemisphere_regions,
			random_state=random_state,
		)
		# mirror symmetry clusters to right hemisphere
		for region_id in np.unique(parcel[parcel.nonzero()]):
			region_id_r = region_id + n_hemisphere_regions * n_layers
			flipped = np.flip(parcel[slices_l], axis=lr_axis)
			parcel[slices_r][flipped == region_id] = region_id_r

			# get center for right cluster
			ctr = cluster_centers[region_id].copy()
			delta = mask.shape[lr_axis] / 2 - ctr[lr_axis]
			ctr[lr_axis] += 2 * delta - 1
			cluster_centers[region_id_r] = ctr

	else:
		atlas_kmeans_l, cluster_centers_l = _do_kmeans(
			x=mask_l,
			num_regions=n_hemisphere_regions,
			random_state=random_state,
		)
		atlas_kmeans_r, cluster_centers_r = _do_kmeans(
			x=mask_r,
			num_regions=n_hemisphere_regions,
			random_state=random_state,
		)
		atlas_kmeans_r += n_hemisphere_regions
		cluster_centers_r = {
			i + n_hemisphere_regions: ctr
			for i, ctr in cluster_centers_r.items()
		}
		parcel = atlas_kmeans_l.copy()
		parcel[mask_r] = atlas_kmeans_r[mask_r]
		cluster_centers = {**cluster_centers_l, **cluster_centers_r}

	regions = parcel[parcel.nonzero()]
	regions = np.unique(regions)
	sklearnex.unpatch_sklearn()

	parc_output = {
		'regions': regions,
		'parcel': parcel.astype('uint32'),
		'cluster_centers': cluster_centers,
		'mask_l': mask_l.astype(bool),
		'mask_r': mask_r.astype(bool),
	}
	return parc_output


def _do_kmeans(x, num_regions, random_state):
	nonzero_voxels = list(zip(*np.where(x.astype(bool))))
	nonzero_voxels = np.reshape(nonzero_voxels, (-1, len(x.shape)))
	nonzero_voxels = nonzero_voxels.astype(int)

	kmeans = KMeans(
		n_clusters=num_regions,
		random_state=random_state,
	).fit(nonzero_voxels.astype(float))
	roi = kmeans.predict(nonzero_voxels.astype(float))

	roi_unique = []
	for lbl in roi:
		if lbl not in roi_unique:
			roi_unique.append(lbl)
	mapping = {lbl: i + 1 for i, lbl in enumerate(roi_unique)}

	roi_relabeld = np.zeros_like(roi)
	for lbl in roi_unique:
		roi_relabeld[roi == lbl] = mapping[lbl]

	atlas_kmeans_dict = collections.defaultdict(list)
	for voxel, lbl in zip(nonzero_voxels, roi_relabeld):
		atlas_kmeans_dict[lbl].append(tuple(voxel))
	atlas_kmeans_dict = dict(atlas_kmeans_dict)

	atlas_kmeans = np.zeros(x.shape, dtype=int)
	for region_id, voxel_list in atlas_kmeans_dict.items():
		atlas_kmeans[tuple(zip(*voxel_list))] = region_id

	cluster_centers = {
		mapping[lbl]: kmeans.cluster_centers_[lbl]
		for lbl in roi_unique
	}
	return atlas_kmeans, cluster_centers


def _compute_dists(id_list, shape, valid):
	points = np.unravel_index(id_list, shape)
	points = np.array(tuple(zip(*points)))
	if valid is not None:
		points = points[:, valid]
	try:
		hull = sp_spat.ConvexHull(points)
		simplices = points[hull.simplices]
		d = sp_dist.cdist(
			XA=points.mean(0, keepdims=True),
			XB=simplices.reshape(-1, points.shape[-1]),
			metric='euclidean',
		)
		return d
	except sp_spat.QhullError:
		return [np.nan]


def infer_radius(
		parcel: np.ndarray,
		radius_fn: Callable = np.nanmedian, ):
	kwargs = {
		'shape': parcel.shape,
		'valid': np.array([
			i > 1 for i in
			parcel.shape
		]),
	}
	with joblib.parallel_backend('multiprocessing'):
		min_dists = joblib.Parallel(n_jobs=-1)(
			joblib.delayed(_compute_dists)
			(id_list, **kwargs) for id_list
			in unique_idxs(parcel).values()
		)
	min_dists = [np.min(e) for e in min_dists]
	return radius_fn(min_dists)


def _erode(id_list, shape, radius, valid=None):
	points = np.unravel_index(id_list, shape)
	points = np.array(tuple(zip(*points)))
	if valid is not None:
		points = points[:, valid]
	try:
		hull = sp_spat.ConvexHull(points)
		if radius == 'boundary':
			remove = hull.simplices.ravel()
			remove = list(set(remove))
			keep = np.delete(range(len(points)), remove)
			points = points[keep]
		else:
			d = sp_dist.cdist(
				XA=points.mean(0, keepdims=True),
				XB=points,
				metric='euclidean',
			).flatten()
			points = points[d < radius]
	except sp_spat.QhullError:
		pass
	return points


def erode_parcellation(
		parcel: np.ndarray,
		radius: Union[float, str] = 'boundary', ):
	if isinstance(radius, str):
		assert radius in ['infer', 'boundary']
	elif not isinstance(radius, float):
		raise ValueError
	if radius == 'infer':
		radius = infer_radius(parcel)
	kwargs = {
		'radius': radius,
		'shape': parcel.shape,
		'valid': np.array([
			i > 1 for i in
			parcel.shape
		]),
	}
	ids = unique_idxs(parcel)
	with joblib.parallel_backend('multiprocessing'):
		points_list = joblib.Parallel(n_jobs=-1)(
			joblib.delayed(_erode)
			(id_list, **kwargs) for
			id_list in ids.values()
		)
	eroded = np.zeros_like(parcel).squeeze()
	for i, points in zip(ids, points_list):
		if not len(points):
			continue
		eroded[tuple(zip(*points))] = i
	eroded = eroded.reshape(parcel.shape).astype('uint32')
	return eroded, radius


def erode_repair(x, radius='boundary'):
	eroded, _ = erode_parcellation(x, radius)
	missing = sorted(
		set(unique_idxs(x)) -
		set(unique_idxs(eroded))
	)
	m = np.isin(x, missing)
	eroded[m] = x[m]
	assert len(unique_idxs(x)) == len(unique_idxs(eroded))
	return eroded


def _fill(id_list, shape, valid):
	points = np.unravel_index(id_list, shape)
	points = np.array(tuple(zip(*points)))
	try:
		tri = sp_spat.Delaunay(points)
		vol = tri.find_simplex(valid) > -1
		return valid[vol]
	except sp_spat.QhullError:
		return


# noinspection PyTypeChecker
def fill_gaps(
		x: np.ndarray,
		valid: np.ndarray = None,
		finalize: bool = False, ):
	if valid is None:
		valid = np.ones(x.shape, dtype=bool)
	ids = unique_idxs(x)
	kwargs = {
		'shape': x.shape,
		'valid': np.array(list(zip(
			*np.where(valid)))),
	}
	with joblib.parallel_backend('multiprocessing'):
		points_list = joblib.Parallel(n_jobs=-1)(
			joblib.delayed(_fill)
			(id_list, **kwargs) for
			id_list in ids.values()
		)
	x_fill = np.zeros_like(x)
	for i, points in zip(ids, points_list):
		if points is None:
			continue
		x_fill[list(zip(*points))] = i

	if finalize:
		x_tmp = x_fill.copy().astype(float)
		x_tmp[~valid] = np.nan
		points = np.where(np.logical_and(
			valid, x_fill == 0))
		points = list(zip(*points))
		for i, j, k in points:
			nn = list(itertools.product(*[
				[e + d for d in [-1, 0, 1]]
				for e in [i, j, k]
			]))
			nn = [e for e in nn if e != (i, j, k)]
			vals = x_tmp[list(zip(*nn))]
			vals = vals[~np.isnan(vals)]
			if not len(vals):
				continue
			vals = collections.Counter(vals).most_common()
			for r, _ in vals:
				if r > 0:
					x_fill[i, j, k] = r
					break
	return x_fill


def fill_img(data: np.ndarray, invalid=None):
	ind = sp_img.distance_transform_edt(
		input=invalid if invalid is
		not None else np.isnan(data),
		return_distances=False,
		return_indices=True,
	)
	return data[tuple(ind)]


def cortex_affiliations(
		save_dir: str,
		affiliation: Dict[str, List[Union[str, int]]] = None,
		as_numpy: bool = True, ):
	affiliation = affiliation if affiliation else {
		'Visual': ['VISpor', 'VISpl', 'VISp', 'VISl', 'VISli', 'VISal', 'VISrl'],
		'Medial': ['RSPv', 'RSPd', 'RSPagl', 'VISpm', 'VISam', 'VISa'],
		'Prefrontal': ['ACAv', 'ACAd', 'PL', 'ILA', 'ORBm', 'ORBvl', 'ORBl', 'FRP'],
		'Somatomotor': [
			'SSp-tr', 'SSp-ll', 'SSp-ul', 'SSp-un', 'SSp-m', 'SSp-n', 'SSp-bfd',
			'SSs', 'MOs', 'MOp'],
		'Lateral': ['TEa', 'ECT', 'PERI', 'VISC', 'GU', 'AIp', 'AId', 'AIv'],
		'Auditory': ['AUDpo', 'AUDp', 'AUDd', 'AUDv'],
	}
	f = pjoin(save_dir, 'obj_top_10um.nii.gz')
	if os.path.isfile(f):
		obj_top = ants.image_read(f, pixeltype='unsigned int')
		f = pjoin(save_dir, 'obj_flat_10um.nii.gz')
		obj_flat = ants.image_read(f, pixeltype='unsigned int')
	else:
		al10 = Allen(10, full_load=False)
		annotation, _ = al10.mcc.get_annotation_volume()
		structs = [s for v in affiliation.values() for s in v]
		structs = al10.get(structs)
		structs = dict(zip(
			structs['id'],
			structs['acronym'],
		))
		# objectmap
		obj = np.zeros_like(annotation)
		for idx, s in structs.items():
			ll_ids = [s + e for e in ['1', '2/3']]
			ll_ids = al10.get(ll_ids)['id'].values
			obj[np.isin(annotation, ll_ids)] = idx
		assert len(np.unique(obj[obj.nonzero()])) == len(structs)
		# top
		obj_top = al10.cortical_map(obj, 'top_view', np.max)
		obj_top = reset_ants_img(obj_top, 10, dtype='uint32')
		ants.image_write(obj_top, f)
		# flat
		obj_flat = al10.cortical_map(obj, 'dorsal_flatmap', np.max)
		obj_flat = reset_ants_img(obj_flat, 10, dtype='uint32')
		f = pjoin(save_dir, 'obj_flat_10um.nii.gz')
		ants.image_write(obj_flat, f)
	if as_numpy:
		obj_top = obj_top.numpy()
		obj_flat = obj_flat.numpy()
	return obj_top, obj_flat, affiliation


def get_node_affiliations(mice):
	obj_top, obj_flat, affiliation = cortex_affiliations()

	aff_top = np.zeros_like(obj_top)
	for i, names in enumerate(affiliation.values()):
		ids = mice.al.get(names)['id'].values
		aff_top[np.isin(obj_top, ids)] = i + 1
	aff_flat = np.zeros_like(obj_flat)
	for i, names in enumerate(affiliation.values()):
		ids = mice.al.get(names)['id'].values
		aff_flat[np.isin(obj_flat, ids)] = i + 1

	cmap, colors = get_affiliation_cmap(return_clist=True)
	colors = {k: c for k, c in zip(affiliation, colors)}

	f = 'parcel2d-flat_10um.nii.gz'
	f = pjoin(mice.cfg.parcel_dir, f)
	parcel_flat = ants.image_read(f, pixeltype='unsigned int')
	parcel_flat = parcel_flat.numpy()

	regions_affil = {}
	for region_id, idxs in unique_idxs(parcel_flat).items():
		percent = dict(collections.Counter(
			aff_flat.flat[idxs]))
		percent = {
			k: percent.get(i, 0) for i, k in
			enumerate(affiliation, start=1)
		}
		percent = {
			k: 100 * np.round(v / sum(percent.values()), 1)
			for k, v in percent.items()
		}
		regions_affil[region_id] = percent
	regions_affil_max = {
		r: max(d.items(), key=lambda t: t[1])[0]
		for r, d in regions_affil.items()
	}
	output = {
		'cmap': cmap,
		'colors': colors,
		'obj_top': obj_top,
		'obj_flat': obj_flat,
		'aff_top': aff_top,
		'aff_flat': aff_flat,
		'affiliation': affiliation,
		'regions_affil': regions_affil,
		'regions_affil_max': regions_affil_max,
	}
	return output
