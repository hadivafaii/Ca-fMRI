from analysis.helper import *
from model.mouse import Mice
from model.configuration import Config
from register.preprocess import (
	remove_badt_ca, butterworth, default_descs)


class Base(object):
	def __init__(
			self,
			mice: Mice = None,
			num_k: int = 7,
			key: str = 'sub-SLC',
			mode: str = 'bold',
			metric: str = 'pearson',
			structs2align: List[str] = None,
			prep_data: bool = False,
			style_defaults: dict = None,
			verbose: bool = False,
			**kwargs,
	):
		super(Base, self).__init__()
		self._setup_style(style_defaults, **kwargs)

		_allowed_modes = ['ca2', 'bold']
		assert mode in _allowed_modes,\
			f"allowed modes: {_allowed_modes}"
		_allowed_metrics = [
			'pearson', 'spearman',
			'cosine', 'correlation',
			'canberra', 'braycurtis',
			'chebyshev', 'minkowski',
			'euclidean', 'seuclidean',
		]
		assert metric in _allowed_metrics,\
			f"allowed metrics: {_allowed_metrics}"
		if mice is None:
			self.mice = Mice(Config())
		else:
			self.mice = mice
		self.num_k = num_k
		self.key = key
		self.mode = mode
		self.task = self.mice.kws.get('task')
		self.metric = metric
		self.verbose = verbose
		self._setup_cmaps()
		self._set_top_flat()
		self._setup_data(prep_data)
		self._setup_lr_labels()
		self._setup_base_name()
		self._setup_struct_ids(
			structs2align)

		self.pi = np_nans((
			self.num_k,
			self.num_n,
		))
		self.pi_fuse = None
		self.contours = {}
		self.affil = None

	def project_vec(
			self,
			vec: np.ndarray = None,
			layer: int = 0,
			proj: str = 'top',
			add_disjoint: bool = False, ):
		vec = vec if vec is not None else self.pi
		vec = _check_2d(vec)

		if add_disjoint:
			disj = self.disjoint_community(vec)
			vec = np.concatenate([vec, np.expand_dims(disj, 0)])

		name = f"parcel2d-{proj}"
		shape = self.mice.parcel[f"shape_{name}"]
		output = np.zeros((len(vec), *shape), dtype=float)
		for node, region_id in self.n2id.items():
			region_id = self.mice.parcel(region_id)
			cond = (
				region_id['layer'] != layer or
				region_id['acro'] != 'Isocortex'
			)
			if cond:
				continue
			reg_k = f"region-idxs_{name}"
			nonzero_k = f"nonzero-idxs_{name}"
			ii = self.mice.parcel.get_id(
				lr=region_id['hemis'],
				cc=region_id['region'],
				ll=0,
			)
			ids = self.mice.parcel[reg_k].get(ii)
			if ids is not None:
				ids = self.mice.parcel[nonzero_k][ids]
				for i, data in enumerate(vec):
					output[i].flat[ids] = data[node]
		return output

	def get_surface_pi(
			self,
			pi: np.ndarray = None,
			agg_fn: Callable = bn.nanmean, ):
		pi = pi if pi is not None else self.pi
		pi = _check_2d(pi)
		pi_surf = np.zeros(pi.shape, dtype=float)
		for lr in [1, 2]:
			for cc in range(1, self.mice.cfg.nn + 1):
				vals = np_nans((self.mice.cfg.ll, len(pi)))
				for ll in range(self.mice.cfg.ll):
					region_id = self.mice.parcel.get_id(
						lr=lr, cc=cc, ll=ll,
					)
					node = self.id2n.get(region_id)
					if node:
						vals[ll] = pi[:, node]
				vals = agg_fn(vals, axis=0)
				id_surf = self.mice.parcel.get_id(
					lr=lr, cc=cc, ll=0,
				)
				node = self.id2n.get(id_surf)
				if node:
					pi_surf[:, node] = vals
		return pi_surf

	def get_struct_fusion_pi(
			self,
			ids: List[List[Union[int, str]]] = None,
			cortex_only: bool = True, ):
		if self.pi_fuse is not None:
			return self.pi_fuse
		ids = ids if ids else self.struct_ids
		masks_fx = self.mice.al.fuse_masks(ids)
		if cortex_only:
			use_nodes = self.mice.parcel(self.id2n)
			use_nodes = use_nodes['acro'] == 'Isocortex'
			use_nodes = list(np.where(use_nodes)[0])
		else:
			use_nodes = list(self.n2id)

		mv_idxs = unique_idxs(self.mice.parcel['brain'])

		fx_idxs = [
			np.where(m.flat)[0]
			for m in masks_fx
		]
		use_i = {
			node: self.n2id[node]
			for node in use_nodes
		}
		with joblib.parallel_backend('multiprocessing'):
			pi_fuse = joblib.Parallel(n_jobs=-1)(
				joblib.delayed(_fuse_score)
				(mv_idxs[use_i[node]], fx_idxs)
				for node in use_nodes
			)
		self.pi_fuse = np.concatenate([
			e.reshape(-1, 1)
			for e in pi_fuse
		], axis=-1)
		return self.pi_fuse

	def align_centroid_to_structs(
			self,
			pi_mv: np.ndarray,
			metric: str = 'cosine',
			cortex_only: bool = True,
			global_order: bool = False,
			assert_equal: bool = False, ) -> List[int]:
		pi_fx = self.get_struct_fusion_pi(cortex_only=cortex_only)
		if cortex_only:
			use_nodes = self.mice.parcel(self.id2n)
			use_nodes = use_nodes['acro'] == 'Isocortex'
			use_nodes = list(np.where(use_nodes)[0])
		else:
			use_nodes = list(self.n2id)
		nonan = np.where(np.isnan(pi_mv).sum(0) == 0)[0]
		use_nodes = set(nonan).intersection(use_nodes)
		use_nodes = sorted(use_nodes)

		if metric in ['cosine', 'correlation']:
			# noinspection PyTypeChecker
			d = sp_dist.cdist(
				XA=pi_fx[:, use_nodes],
				XB=pi_mv[:, use_nodes],
				metric=metric,
			)
			if np.isnan(d).sum():
				# TODO: handle this using logger
				# print('Warning, nan in global order')
				d[np.isnan(d)] = 1.
			scores = dist2sim(d)
			mappings = find_best_match(
				scores=scores,
				reverse=True,
				global_order=global_order,
				assert_equal=assert_equal,
			)
			gmap = list(mappings.values())

		else:
			masks_mv = self.disjoint_masks_from_pi(
				pi=pi_mv, cortex_only=cortex_only)
			masks_fx = self.mice.al.fuse_masks(self.struct_ids)
			masks_fx = {i: m for i, m in enumerate(masks_fx)}
			_, mappings = clusters_sim_scores(
				masks_fx=masks_fx,
				masks_mv=masks_mv,
				global_order=global_order,
				assert_equal=assert_equal,
			)
			gmap = list(mappings[metric].values())
		if len(gmap) < len(pi_mv):
			gmap += list(np.delete(range(len(pi_mv)), gmap))
		return gmap

	def disjoint_masks_from_pi(self, pi, cortex_only: bool = True):
		community = self.disjoint_community(pi)
		masks = self.community2masks(community)
		if cortex_only:
			masks = {
				i: np.logical_and(
					m, self.mice.al.get_masks('Isocortex')
				) for i, m in masks.items()
			}
		# noinspection PyTypeChecker
		masks = dict(sorted(masks.items()))
		return masks

	def disjoint_community(
			self,
			pi: np.ndarray = None,
			to_dict: bool = False, ):
		pi = pi if pi is not None else self.pi
		partition = 1.0 + np.argmax(pi, axis=0)
		nans = np.isnan(pi).sum(0) > 0
		partition[nans] = np.nan
		if to_dict:
			partition = {
				i: list(itertools.compress(
					data=self.id2n,
					selectors=partition == i,
				)) for i in sorted(set(partition))
			}
		return partition

	def community2masks(self, community: Dict[int, List[str]]):
		cluster_masks = {}
		for c_idx, labels in community.items():
			x = np.zeros(self.mice.al.dims)
			for lbl in labels:
				if len(lbl.split('-')) == 3:
					ii = self.mice.parcel.encode(lbl)
					_m = self.mice.parcel['cortex'] == ii
				else:
					lr, ii = tuple(map(lambda s: int(s), lbl.split('-')))
					_m = self.mice.parcel['subcortex'] == ii
					if lr == 1:
						_m = np.logical_and(_m, self.mice.al.mask_l)
					else:
						_m = np.logical_and(_m, self.mice.al.mask_r)
				x[_m] = c_idx
			cluster_masks[c_idx] = x.astype(bool)
		return cluster_masks

	def to_vol(self, x: Union[dict, np.ndarray], cortex_only: bool = False):
		if isinstance(x, np.ndarray):
			x = {i + 1: x[i] for i in range(len(x))}
		x_vec = {
			k: {
				region_id: v[n]
				for n, region_id
				in self.n2id.items()
			}
			for k, v in x.items()
		}
		x_vol = vec2vol(x_vec, self.mice)
		if cortex_only:
			x_vol = {
				idx: v * self.mice.al.iso_mask
				for idx, v in x_vol.items()
			}
		return x_vol

	def symmetry_score(
			self,
			pi: np.ndarray = None,
			metric: str = 'correlation',
			partition: np.ndarray = None,
			decimals: int = 3, ):
		def _finalize(x):
			x = {**x, -1: np.mean(list(x.values()))}
			x = {
				idx: np.round(s, decimals)
				for idx, s in x.items()
			}
			return x

		pi = pi if pi is not None else self.pi
		pi = _check_2d(pi)
		# Overlapping
		nans = np.isnan(pi).sum(0) > 0
		nans = np.logical_or(nans[self.l_idxs], nans[self.r_idxs])
		nonan = np.where(~nans)[0]
		# noinspection PyTypeChecker
		scores_ovp = dist2sim(sp_dist.cdist(
			XA=pi[:, self.l_idxs][:, nonan],
			XB=pi[:, self.r_idxs][:, nonan],
			metric=metric,
		)).diagonal()
		scores_ovp = {
			i + 1: s for i, s in
			enumerate(scores_ovp)
		}
		# Disjoint
		if partition is None:
			partition = self.disjoint_community(pi)
		scores_disj = {
			idx: 1 - sp_dist.hamming(
				u=partition[self.l_idxs][nonan] == idx,
				v=partition[self.r_idxs][nonan] == idx,
			) for idx in range(1, 1 + self.num_k)
		}
		return _finalize(scores_ovp), _finalize(scores_disj)

	def get_inter_intra(self, a: np.ndarray):
		if a.shape != (self.num_n,) * 2:
			l_idxs = range(len(a) // 2)
			r_idxs = range(len(a) // 2, len(a))
		else:
			l_idxs = self.l_idxs
			r_idxs = self.r_idxs
		inter = a[l_idxs][:, r_idxs]
		intra_l = a[l_idxs][:, l_idxs]
		intra_r = a[r_idxs][:, r_idxs]
		intra = intra_l.copy()
		upper = np.triu_indices_from(intra)
		intra[upper] = intra_r[upper]
		np.fill_diagonal(intra, 1.0)
		return inter, intra

	def node_affiliations(self):
		if self.affil is not None:
			return self.affil
		from register.parcellation import get_node_affiliations
		self.affil = get_node_affiliations(self.mice)
		return self.affil

	def get_contour(
			self,
			width: float = 0.3,
			proj: str = 'top',
			coarse: bool = True,
			force: bool = False, ):
		if proj in self.contours and not force:
			return self.contours[proj]
		if self.affil is None:
			_ = self.node_affiliations()
		obj = self.affil[f"obj_{proj}"].copy()

		if coarse:
			subtree, _ = self.mice.al.get_subtree(315, 2, False)
			children = collections.defaultdict(list)
			nodes = [n.identifier for n in subtree.children(315)]
			nodes += [self.mice.al.get('SSp')['id']]
			for node in nodes:
				for child in subtree.children(node):
					children[node].append(child.identifier)

			for node, child_list in children.items():
				obj[np.isin(obj, child_list)] = node
		contour = get_contour(obj, 0.125, width)
		contour = self.trim2d(contour)[0]
		self.contours[proj] = contour
		return contour

	def _setup_struct_ids(self, structs2align):
		s = structs2align if structs2align else [
			['ACA', 'PL'],
			['VISp', 'VISpm'],
			['SSp-m', 'SS-n', 'SSs'],
			['SSp-ll', 'SSp-tr'],
			['AId', 'AIv'],  # TODO: this is invisible in 2D
			['ECT', 'PERI', 'TEa'],
		]
		self.struct_ids = [
			self.mice.al.get(e)['id'].tolist()
			for e in s
		]
		return

	def _setup_data(self, prepare_data: bool):
		if prepare_data:
			proc = self.mice.get_data_containers(
				mode=self.mode, force=False)[0]
			data = proc.get(self.key)
			if self.task is None:
				self.task = data.task
			assert self.task == data.task
		else:
			data = None

		if self.mode in ['ca2', 'bold']:
			self.n2id = self.mice.node_lookup[self.mode]
			if prepare_data:
				data = data[:]
		else:
			raise ValueError(f"invalid mode: '{self.mode}'")

		self.id2n = {r: i for i, r in self.n2id.items()}
		self.num_n = len(self.n2id)

		if prepare_data and self.mode == 'ca2':
			data = self._process_ca(data)
		self.data = data
		if self.data is not None:
			assert len(self.data) == self.num_n
		return

	def _process_ca(self, data: np.ndarray, n_std: float = 5.0):
		fs = self.mice.cfg.fs_ca
		# take out first 50s
		data, ab, bad_t = remove_badt_ca(
			data=data,
			run_duration=self.mice.cfg.run_duration,
			exclude=self.mice.cfg.exclude,
			n_std=n_std,
			fs=fs,
		)
		# apply bandpass filter
		b = self.mice.kws.get(f"band_{self.mode}")
		if b is not None:
			assert isinstance(b, tuple) and len(b) == 2
			if min(b) > 0 or max(b) < fs / 2:
				data = butterworth(data, b, fs, 3)
		# apply frame censoring
		df = pjoin(self.mice.cfg.base_dir, 'meta_info')
		df = pd.read_pickle(pjoin(df, 'tmask_hp.df'))
		if self.key in df['key'].unique():
			tmask = df.loc[
				df['key'] == self.key,
				'temporal_mask',
			].values.astype(bool)
			tmask = np.repeat(tmask, fs)
			tmask = tmask[ab]
			if bad_t is not None:
				tmask = np.delete(
					tmask, bad_t, axis=0)
			assert data.shape[1] == len(tmask)
			data = data[:, tmask]
		# if eps is not None:
		# data[bn.nanvar(data, axis=1) < eps] = np.nan
		return data

	def _setup_lr_labels(self):
		self.l_idxs = [
			n for n, i in self.n2id.items() if
			self.mice.parcel(i)['hemis'] == 1
		]
		self.r_idxs = [
			n for n, i in self.n2id.items() if
			self.mice.parcel(i)['hemis'] == 2
		]
		return

	def _setup_cmaps(self):
		cmap_all, colors = get_cluster_cmap(
			color_ids=range(self.num_k),
			return_colors=True,
		)
		cmaps = {
			i + 1: get_cluster_cmap(i)
			for i in range(self.num_k)
		}
		self.cmaps = {**cmaps, -1: cmap_all}
		self.colors = {i + 1: c for i, c in enumerate(colors)}
		mrk = ['o', 'X', 'P', 's', 'D', 'p', 'v', '^', '<', '>']
		self.markers = mrk[:self.num_k]
		return

	def _setup_style(self, defaults, **kwargs):
		defaults = defaults if defaults else {}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v
		self.style = kwargs

	def _setup_base_name(self):
		mode = self.mode.split('-')[0]
		band = self.mice.kws.get(f"band_{mode}")
		desc = self.mice.kws.get(f'desc_{mode}')
		assert desc in default_descs(mode)
		name = [
			f"n-{self.mice.cfg.nn}"
			if self.mice.cfg.parcel_mode == 'spatial'
			else f"n-{self.mice.cfg.nn}*{self.mice.cfg.ll}",
			self.metric,
			self.task,
			'-'.join([self.mode, desc]),
			f"bp{band}".replace(' ', '')
			if band else None,
			self.key,
		]
		name = [e for e in name if e]
		name = '_'.join(name)
		self.base_name = name
		return

	def name(self):
		return self.base_name

	def path(self, mkdir: bool = False):
		path = [
			self.mice.cfg.results_dir,
			self.__class__.__name__,
			self.name(),
		]
		path = pjoin(*path)
		if mkdir:
			os.makedirs(path, exist_ok=True)
		return path

	def load_top_flat(self):
		topview = 'topview_10um.nii.gz'
		topview = pjoin(self.mice.cfg.base_dir, topview)
		topview = ants.image_read(topview, pixeltype='unsigned int')
		topview = topview.numpy()
		flatmap = 'flatmap_10um.nii.gz'
		flatmap = pjoin(self.mice.cfg.base_dir, flatmap)
		flatmap = ants.image_read(flatmap, pixeltype='unsigned int')
		flatmap = flatmap.numpy()
		return topview, flatmap

	def _set_top_flat(self):
		topview, flatmap = self.load_top_flat()
		self.topview, self.flatmap = topview, flatmap
		return

	def trim2d(self, data: np.ndarray):
		if not (hasattr(self, 'topview') and hasattr(self, 'flatmap')):
			self._set_top_flat()
		if data.ndim == 2:
			is_2d = True
			data = np.expand_dims(data, 0)
		elif data.ndim == 3:
			is_2d = False
		else:
			raise RuntimeError(f"Invalid ndim: {data.ndim}")
		if data.shape[1:] == self.topview.shape:
			bg = self.topview
		elif data.shape[1:] == self.flatmap.shape:
			bg = self.flatmap
		else:
			raise RuntimeError(f"Invalid data shape: {data.shape}")
		results = filter_boundaries([bg] + list(data), bg > 0)
		bg = results.pop(0)
		x_tr = np.concatenate([
			np.expand_dims(e, 0)
			for e in results
		])
		if is_2d:
			x_tr = x_tr[0]
		return x_tr, bg

	def show(
			self,
			data: dict = None,
			vmin: float = 0.1,
			vmax: float = 1.0,
			proj: str = 'top',
			cbar: bool = False,
			contour: bool = True,
			add_ylabels: bool = True,
			titles: List[str] = False,
			mask_thresh: float = None,
			display: bool = True,
			**kwargs, ):
		defaults = {
			'dpi': 100,
			'layout': None,
			'wspace': None,
			'hspace': None,
			'figsize': None,
			'figsize_x': 5.0,
			'figsize_y': 1.7,
			'figsize_scale': 5,
			'interp': 'antialiased',
			'brain_cmap': 'Greys_r',
			'contour_cmap': 'Greys',
			'contour_alpha': 0.8,
			'label_fontsize': 13,
			'label_pad': 15,
			'title_y': 1.06,
			'cmap': 'magma',
		}
		for k, v in defaults.items():
			if k not in kwargs:
				kwargs[k] = v
		kws_proj = {
			'proj': proj,
			'layer': kwargs.get('layer', 0),
			'add_disjoint': kwargs.get('add_disjoint', False),
		}
		kws_contour = {
			'proj': proj,
			'width': kwargs.get('contour_width', 0.2),
			'coarse': kwargs.get('contour_coarse', True),
		}
		kws_cbar = {
			'pad': kwargs.get('cbar_pad', 0.05),
			'aspect': kwargs.get('cbar_aspect', 20),
			'shrink': kwargs.get('cbar_shrink', 1.0),
			'fraction': kwargs.get('cbar_fraction', 0.15),
			'location': kwargs.get('cbar_location', 'right'),
		}
		if contour:
			contour = self.get_contour(**kws_contour)
			contour = mwh(~contour, contour)
		else:
			contour = None
		# handle masking threshold
		if mask_thresh is None:
			mask = np.ones(self.num_n)
			mask = self.trim2d(self.project_vec(
				vec=mask, **kws_proj))[0][0]
			mask = mask == 1
		elif isinstance(mask_thresh, float):
			mask = None
		else:
			msg = f"invalid mask threshold:\n'{mask_thresh}'"
			raise ValueError(msg)
		# handle data
		if data is None:
			data = {'pi': self.trim2d(self.project_vec(
				**kws_proj))}
		elif isinstance(data, np.ndarray):
			data = {'': data}
		if isinstance(data, dict):
			if any(isinstance(x, np.ndarray) for x in data.values()):
				data = {
					lbl: self.trim2d(self.project_vec(
						vec=x, **kws_proj,
					)) for lbl, x in data.items()
				}
		elif isinstance(data, (list, tuple)):
			data = {
				i: self.trim2d(self.project_vec(
					vec=x, **kws_proj,
				)) for i, x in enumerate(data)
			}
		else:
			msg = f"invalid data type: '{type(data).__name__}'"
			raise RuntimeError(msg)
		# create figure
		ncols = max(
			len(tr[0]) for tr
			in data.values()
		)
		scale = ncols / kwargs['figsize_scale']
		scale = np.maximum(1.0, scale)
		if kwargs['figsize'] is None:
			kwargs['figsize'] = (
				kwargs['figsize_x'] * scale * 3,
				kwargs['figsize_y'] * scale * len(data),
			)
			kwargs['label_fontsize'] *= scale
		fig, axes = create_figure(
			nrows=len(data),
			ncols=ncols,
			tight_layout=False,
			figsize=kwargs['figsize'],
			layout=kwargs['layout'],
			wspace=kwargs['wspace'],
			hspace=kwargs['hspace'],
			dpi=kwargs['dpi'],
			reshape=True,
		)
		# handle titles
		if titles is True:
			titles = [
				f"OC-{i}" for i in
				range(1, ncols + 1)
			]
			if kws_proj['add_disjoint']:
				titles = titles[:-1]
				titles += ['Disj']
		if titles is not False:
			if not isinstance(titles, list):
				titles = list(titles)
			titles += [''] * (ncols - len(titles))
		# do plotting
		for i, (lbl, (x, bg)) in enumerate(data.items()):
			for j in range(ncols):
				try:
					ax = axes[i, j]
				except TypeError:
					ax = axes
				if j >= len(x):
					ax.remove()
				else:
					ax.imshow(
						X=mwh(bg == 0, bg),
						cmap=kwargs['brain_cmap'],
					)
					cond = (
						kws_proj['add_disjoint'] and
						np.nanmax(x[j]) > 1
					)
					if cond:
						c_ids = range(len(x) - 1)
						kws_im = {
							'interpolation': 'none',
							'cmap': get_cluster_cmap(c_ids),
							'vmin': 1,
							'vmax': c_ids.stop,
						}
					else:
						kws_im = {
							'interpolation': kwargs['interp'],
							'cmap': kwargs['cmap'],
							'vmin': vmin,
							'vmax': vmax,
						}
					x2p = x[j]
					if mask is None:
						x2p = mwh(x2p < mask_thresh, x2p)
					else:
						x2p = mwh(~mask, x2p)
					im = ax.imshow(x2p, **kws_im)
					if cbar and j == ncols - 1:
						plt.colorbar(im, ax=ax, **kws_cbar)
					if contour is not None:
						ax.imshow(
							X=contour,
							cmap=kwargs['contour_cmap'],
							alpha=kwargs['contour_alpha'],
						)
					if add_ylabels and j == 0:
						ax.set_ylabel(
							ylabel=lbl,
							fontsize=kwargs['label_fontsize'],
							labelpad=kwargs['label_pad'],
						)
					if titles and i == 0:
						ax.set_title(
							label=titles[j],
							y=kwargs['title_y'],
							fontsize=kwargs['label_fontsize'],
						)
		remove_ticks(axes)
		if display:
			plt.show()
		else:
			plt.close()
		return fig, axes


def _fuse_score(ids, fx_idxs):
	scores = np.zeros(len(fx_idxs))
	for k, ids_fx in enumerate(fx_idxs):
		s = len(set(ids).intersection(ids_fx))
		s /= min(len(ids), len(ids_fx))
		scores[k] = s
	return scores


def _check_2d(x):
	if x.ndim == 1:
		return np.expand_dims(x, 0)
	elif x.ndim == 2:
		return x
	else:
		return avg(x)
