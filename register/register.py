from utils.generic import *
from utils.plotting import create_figure, plt
from analysis.helper import symmetry_score, overlap_score
from .preprocess import compute_tsnr_mask, default_descs


class Register(object):
	def __init__(
			self,
			mice,
			in_house_res: int = 200,
			in_house_name: str = 'Symmetric_N162',
			tr_axes: Tuple[int] = (2, 1, 0),
			verbose: bool = True,
			space: str = 'CCF',
	):
		super(object, self).__init__()
		self.mice = mice
		self.mice.setup_anat_data()
		self.in_house_res = in_house_res
		self.in_house_name = in_house_name
		self.tr_axes = tr_axes
		self.verbose = verbose
		self.space = space

		in_house = self.load_in_house(in_house_res)
		self.in_house = self.reorient(in_house)

		if self.mice.al.template is None:
			self.mice.al.load_everything(self.mice.cfg.resolution)
		self.template = self.img_in_ccf(self.mice.al.template, float)
		self.nissl = self.img_in_ccf(self.mice.al.nissl, float)
		self.objects = self.get_objects()
		self.roi_thres = 0.2
		self.roi_cutoff = 0
		self.mi_base = None
		self.mode = 'bold'
		self.dim = '3d'

	def set_mode(self, mode):
		assert mode in ['bold', 'ca2']
		self.mode = mode
		self.dim = '2d' if mode == 'ca2' else '3d'

	def get_objects(self, structs=None):
		structs = structs if structs else [
			'VIS', 'TEa', 'PTLp',
			'SS', 'MOs', 'MOp', 'VISC',
			'AI', 'STRd', 'CA1', 'CLA',
		]
		objects = {
			'misc': self.mice.al.get_masks(structs, True),
			'iso': self.mice.al.get_masks('Isocortex', True),
			'hemis': self.mice.al.make_hemis(),
		}
		objects = {
			k: self.img_in_ccf(v)
			for k, v in objects.items()
		}
		return objects

	def acro2inhouse(
			self,
			name: str = None,
			fixed: ants.ANTsImage = None,
			moving: ants.ANTsImage = None,
			structs: List[str] = None, ):
		if structs is None:
			s = self.mice.al.get_major_structs()
			idx = s.index('STR') + 1
			structs = s[:idx] + ['Isocortex'] + s[idx:]
		objects = {
			s: self.mice.al.get_masks(s)
			for s in structs
		}
		objects = {
			**objects,
			'hemis': self.mice.al.make_hemis(),
		}
		objects = {
			k: self.img_in_ccf(v)
			for k, v in objects.items()
		}

		name = name if name else self.in_house_name
		fixed = fixed if fixed else self.nissl
		moving = moving if moving else self.in_house
		tx, tx_aff = self.load_tx(name)
		affined = tx_aff.apply_to_image(
			image=moving,
			reference=fixed,
			interpolation='linear',
		)
		obj = self.apply2objects(
			obj_fixed=objects,
			moving=affined,
			moving_orig=moving,
			tx_aff=tx_aff,
			tx=tx,
		)
		obj = {
			k: self.to_lps(v)
			for k, v in obj.items()
		}
		return obj, obj.pop('hemis')

	def apply2objects(
			self,
			moving: ants.ANTsImage,
			tx: Union[List[str], Dict[str, list]],
			obj_fixed: dict = None,
			moving_orig: ants.ANTsImage = None,
			tx_aff: ants.ANTsTransform = None, ):
		if obj_fixed is None:
			obj_fixed = self.objects
		if isinstance(tx, list):
			tx = {'invtransforms': tx}
		obj = {
			k: ants.apply_transforms(
				fixed=moving,
				moving=img,
				transformlist=tx['invtransforms'],
				interpolator='genericLabel',
			) for k, img in obj_fixed.items()
		}
		if tx_aff is not None:
			inv = tx_aff.invert()
			obj = {
				k: inv.apply_to_image(
					image=img,
					reference=moving_orig,
					interpolation='multilabel',
				) for k, img in obj.items()
			}
		obj = {
			k: reset_ants_img(
				img=img,
				origin=moving_orig.origin if
				moving_orig else moving.origin,
				spacing=moving_orig.spacing if
				moving_orig else moving.spacing,
				dtype='uint32',
			) for k, img in obj.items()
		}
		return obj

	def organize_func(
			self,
			cutoff: int = 0,
			thres: float = 0.2,
			q_bold: float = 0.05, ):
		logger = make_logger(
			name='organize_func',
			path=self.mice.cfg.log_dir,
			verbose=self.verbose,
			module=__name__,
		)
		lookup = {}
		self.roi_thres = thres
		self.roi_cutoff = cutoff
		tsnr_masks = self.get_tsnr_masks(
			q_bold=q_bold)
		for mode in ['ca2', 'bold']:
			self.set_mode(mode)
			accepted = self.get_accepted_regions()
			accepted = accepted['acc']
			lookup[mode] = {
				i: region for i, region
				in enumerate(accepted)
			}
			for desc in default_descs(mode):
				kws = {f"desc_{mode}": desc}
				self.mice.setup_preproc_data(**kws)
				self._organize_func(
					tsnr_masks=tsnr_masks[mode],
					accepted_regions=accepted,
					description=desc,
					logger=logger,
				)
		save_obj(
			obj=lookup,
			file_name='roi_lookup',
			save_dir=self.mice.cfg.main_dir,
			verbose=self.verbose,
			mode='npy',
		)
		return

	def save_png(
			self,
			name: str,
			fixed: ants.ANTsImage,
			objects: Dict[str, ants.ANTsImage],	):
		path = pjoin(self.mice.cfg.masks_dir, 'png', name)
		os.makedirs(path, exist_ok=True)
		for k, img in objects.items():
			if k != 'iso':
				continue
			ax_names = {0: 'cor', 1: 'axi', 2: 'sag'}
			for axis, axis_name in ax_names.items():
				file = [
					name,
					f"space-indiv{self.dim}",
					f"{k}-inv-{axis_name}",
				]
				file = f"{'_'.join(file)}.png"
				file = pjoin(path, file)
				if len(np.unique(img.numpy())) <= 2:
					cmap = 'Reds'
				else:
					cmap = 'tab10'
				fixed.plot(
					overlay=img,
					axis=axis,
					overlay_alpha=0.5,
					overlay_cmap=cmap,
					nslices=49,
					figsize=1,
					filename=file,
				)
		return

	def register_anat(
			self,
			save_results: bool,
			anchor_ses: int = None,
			fixed: ants.ANTsImage = None,
			**kwargs, ):
		self.set_mode('bold')
		self.mi_base = None
		fixed = fixed if fixed else self.nissl

		# in house
		tx, tx_aff = self.load_tx(self.in_house_name)
		if tx:
			affined = tx_aff.apply_to_image(
				image=self.in_house,
				reference=fixed,
				interpolation='linear',
			)
			_ = self._apply_tx_obj2lps(
				name=self.in_house_name,
				moving=affined,
				moving_orig=self.in_house,
				tx_aff=tx_aff,
				tx=tx,
				use_mm=True,
				save_results=save_results,
			)
		else:
			_ = self.fit_register(
				name=self.in_house_name,
				moving=self.in_house,
				fixed=fixed,
				save_results=True,
				**kwargs,
			)

		# indivs
		_s = []
		obj_all = {}
		if anchor_ses is None:
			pbar = tqdm(
				self.mice.T1w.items(),
				total=len(self.mice.T1w),
				dynamic_ncols=True,
				leave=self.verbose,
				position=0,
			)
			for key, anat in pbar:
				pbar.set_description(f"registering anat {key}")
				if anat.nii_file is None:
					continue
				indiv = anat.load_ants()
				indiv *= self.mice.mask3d[key].load_ants()
				indiv = self.reorient(indiv)
				tx, tx_aff = self.load_tx(key)
				if tx:
					affined = tx_aff.apply_to_image(
						image=indiv,
						reference=fixed,
						interpolation='linear',
					)
					obj_all[key] = self._apply_tx_obj2lps(
						name=key,
						moving=affined,
						moving_orig=indiv,
						tx_aff=tx_aff,
						tx=tx,
						use_mm=True,
						save_results=False,
					)
				else:
					results = self.fit_register(
						name=key,
						moving=indiv,
						fixed=fixed,
						save_results=True,
						**kwargs,
					)
					if results:
						_s.append(results['perf'])
		else:
			other_ses = [
				j for j in self.mice.cfg.ses_ids
				if j != anchor_ses
			]
			for sub in tqdm(self.mice.cfg.sub_ids):
				key_fx = self.mice.get_key(sub, anchor_ses)
				fx = self.mice.T1w[key_fx].load_ants()
				fx *= self.mice.mask3d[key_fx].load_ants()
				fx_prep = self.reorient(fx)

				# Load or perform CCF --> Selected ses
				tx, tx_aff = self.load_tx(key_fx)
				if not tx:
					results = self.fit_register(
						name=key_fx,
						moving=fx_prep,
						fixed=fixed,
						save_results=True,
						**kwargs,
					)
					if results:
						_s.append(results['perf'])
				tx, tx_aff = self.load_tx(key_fx)

				affined = tx_aff.apply_to_image(
					image=fx_prep,
					reference=fixed,
					interpolation='linear',
				)
				obj = self._apply_tx_obj2lps(
					name=key_fx,
					moving=affined,
					moving_orig=fx_prep,
					tx_aff=tx_aff,
					tx=tx,
					use_mm=True,
					save_results=False,
				)
				obj_all[key_fx] = obj

				for ses_mv in other_ses:
					key_mv = self.mice.get_key(sub, ses_mv)
					mv = self.mice.T1w[key_mv].load_ants()
					mv *= self.mice.mask3d[key_mv].load_ants()

					tx_rig = ants.registration(
						fixed=mv,
						moving=fx,
						type_of_transform='DenseRigid',
						verbose=False,
					)
					obj_all[key_mv] = {
						k: ants.apply_transforms(
							fixed=mv,
							moving=o,
							transformlist=tx_rig['fwdtransforms'],
							interpolator='genericLabel',
						) for k, o in obj.items()
					}
		# override erroded (rigid removes some ROIs)
		for k, v in obj_all.items():
			v['eroded'] = _erode_post(v['brain'])
		obj_all = dict(sorted(obj_all.items()))

		if save_results:
			for name, obj in obj_all.items():
				for label, img in obj.items():
					file = [
						name,
						f"space-indiv{self.dim}",
						label,
					]
					file = f"{'_'.join(file)}.nii.gz"
					file = pjoin(self.mice.cfg.masks_dir, file)
					ants.image_write(img, file)

		if self.verbose and _s:
			msg = 'overal scores:\t'
			msg += f"{np.mean(_s):.3f} ± {np.std(_s):.3f}\n\n"
			print(msg)
		return obj_all

	def tx_peformance(
			self,
			tx: Dict[str, list],
			fixed: ants.ANTsImage,
			moving: ants.ANTsImage,
			obj_fixed: dict = None,
			moving_orig: ants.ANTsImage = None,
			tx_aff: ants.ANTsTransform = None, ):

		scores = {}
		warped = tx.get('warpedmovout', None)
		warped = warped if warped else ants.apply_transforms(
			fixed=fixed,
			moving=moving,
			transformlist=tx['fwdtransforms'],
			interpolator='linear',
		)
		try:
			mi = ants.image_mutual_information(
				image1=fixed, image2=warped)
			mi /= self.mi_base
			scores.update({'mi': mi})
		except RuntimeError:
			scores.update({'mi': 0.0})
			return scores

		obj_fixed = obj_fixed if obj_fixed else self.objects
		obj = self.apply2objects(
			tx=tx,
			tx_aff=tx_aff,
			moving=moving,
			moving_orig=moving_orig,
			obj_fixed=obj_fixed,
		)
		if tx_aff is not None:
			m = moving_orig.get_mask().numpy().astype(bool)
		else:
			m = moving.get_mask().numpy().astype(bool)
		m_l = obj['hemis'].numpy() == 1
		m_r = obj['hemis'].numpy() == 2

		misc = np.unique(self.objects['misc'].numpy())
		misc = [int(e) for e in misc if e > 0]
		misc = self.mice.al.get(misc)
		misc = dict(zip(misc['acronym'], misc['id']))
		misc = {
			k: obj['misc'].numpy() == i
			for k, i in misc.items()
		}
		misc['iso'] = obj['iso'].numpy().astype(bool)

		for key, x in misc.items():
			scores[f"ovp_{key}"] = overlap_score(x, m)
			scores[f"sym_{key}"] = symmetry_score(x, m_l, m_r)
		scores = {
			k: 0.0 if np.isnan(s) else s
			for k, s in scores.items()
		}

		_ = misc.pop('iso')

		vox = self.mice.al.get_masks(list(misc), False)
		vox = {k: v.astype(bool).sum() for k, v in vox.items()}
		sym, ovp = 0, 0
		for key, size in vox.items():
			sym += size * scores[f"sym_{key}"]
			ovp += size * scores[f"ovp_{key}"]
		scores.update({
			'sym': sym / sum(vox.values()),
			'ovp': ovp / sum(vox.values()),
		})
		return scores, misc, warped

	def _fit_aff_init(self, name, fixed, moving, save_results):
		perf = {}
		aff_all = []
		mi = 0.0
		_, tx_aff = self.load_tx(name)
		if tx_aff is not None:
			moving_aff = tx_aff.apply_to_image(
				image=moving,
				reference=fixed,
				interpolation='linear',
			)
			mi = ants.image_mutual_information(
				image1=moving_aff, image2=fixed)
			mi /= self.mi_base
			if mi > 0.4:
				perf[f"loaded"] = mi
				aff_all.append((None, tx_aff, moving_aff))
				save_results = False
		looper = itertools.product([5, 10], [0.1, 0.2])
		for fac, rad in looper:
			if mi > 0.40:
				break
			txfile = ants.affine_initializer(
				fixed_image=fixed,
				moving_image=moving,
				search_factor=fac,
				radian_fraction=rad,
				use_principal_axis=False,
				local_search_iterations=10,
			)
			tx_aff = ants.read_transform(txfile)
			moving_aff = tx_aff.apply_to_image(
				image=moving,
				reference=fixed,
				interpolation='linear',
			)
			mi = ants.image_mutual_information(
				image1=moving_aff, image2=fixed)
			mi /= self.mi_base
			perf[f"fac: {fac:d}, rad: {rad:.2f}"] = mi
			aff_all.append((txfile, tx_aff, moving_aff))

		best_idx = np.argmax(list(perf.values()))
		txfile, tx_aff, moving_aff = aff_all[best_idx]

		if self.verbose:
			fig, axes = create_figure(1, 2, (8, 3.1))
			slice_i = int(0.57 * fixed.shape[0])
			axes[0].imshow(fixed[slice_i], cmap='Greys_r')
			axes[1].imshow(moving_aff[slice_i], cmap='Greys_r')
			msg = f"{name},  (winner)  {list(perf)[best_idx]}, "
			msg += f"perf: {list(perf.values())[best_idx]:0.3f}"
			fig.suptitle(msg, fontsize=15)
			plt.show()

		if save_results:
			dst = [
				name,
				f"space-{self.space}{self.dim}",
				'ants-aff',
			]
			dst = '_'.join(dst)
			ext = '.'.join(txfile.split('.')[1:])
			dst = f"{dst}.{ext}"
			dst = pjoin(self.mice.cfg.tx_dir, dst)
			shutil.copyfile(src=txfile, dst=dst)

		return tx_aff, moving_aff

	def fit_register(
			self,
			name: str,
			fixed: ants.ANTsImage,
			moving: ants.ANTsImage,
			logger: logging.Logger = None,
			affine_init: bool = True,
			save_results: bool = False,
			syn_metric: List[str] = None,
			aff_metric: List[str] = None,
			types: List[str] = None,
			grads: List[float] = None,
			syn_bins: List[int] = None,
			aff_bins: List[int] = None,
			aff_rates: List[int] = None,
			mi_min: float = 0.4,
			mi_q: float = 0.75,
			n_seeds: int = 10, ) -> dict:

		logger = logger if logger else make_logger(
			name=f"register_anat_{name}",
			path=self.mice.cfg.log_dir,
			module=__name__,
			verbose=False,
		)
		syn_metric = syn_metric if syn_metric else ['mattes']
		aff_metric = aff_metric if aff_metric else ['mattes']
		types = types if types else ['SyN', 'ElasticSyN']
		grads = grads if grads else [0.2, 0.3]
		syn_bins = syn_bins if syn_bins else [32]
		aff_bins = aff_bins if aff_bins else [13, 32]
		aff_rates = aff_rates if aff_rates else [0.2, 0.8]

		if self.mi_base is None:
			self.mi_base = ants.image_mutual_information(
				image1=fixed, image2=fixed)

		if affine_init:
			tx_aff, moving_aff = self._fit_aff_init(
				name=name,
				fixed=fixed,
				moving=moving,
				save_results=save_results,
			)
		else:
			tx_aff, moving_aff = None, None

		dlist = []
		tx_all = {}
		for seed in range(n_seeds):
			looper = itertools.product(
				syn_metric, aff_metric,
				types, grads, syn_bins,
				aff_bins, aff_rates)
			for _sm, _am, _t, _g, _syn, _aff, _r in looper:
				tx = ants.registration(
					fixed=fixed,
					moving=moving_aff if
					moving_aff else moving,
					aff_metric=_am,
					syn_metric=_sm,
					type_of_transform=_t,
					grad_step=_g,
					aff_sampling=_aff,
					aff_random_sampling_rate=_r,
					syn_sampling=_syn,
					random_seed=seed,
					verbose=False,
				)
				s, _, _ = self.tx_peformance(
					fixed=fixed,
					moving=moving_aff if
					moving_aff else moving,
					moving_orig=moving if
					moving_aff else None,
					tx_aff=tx_aff,
					tx=tx,
				)
				if s['mi'] < mi_min:
					continue
				dlist.append({
					'seed': [seed] * len(s),
					'syn_metric': [_sm] * len(s),
					'aff_metric': [_am] * len(s),
					'type': [_t] * len(s),
					'grad': [_g] * len(s),
					'syn_bin': [_syn] * len(s),
					'aff_bin': [_aff] * len(s),
					'aff_rate': [_r] * len(s),
					'metric': s.keys(),
					'score': s.values(),
				})
				tx_id = _mk_tx_string(
					seed=seed,
					sm=_sm,
					am=_am,
					t=_t,
					g=_g,
					syn=_syn,
					aff=_aff,
					r=_r,
				)
				tx_all[tx_id] = {
					k: v for k, v in tx.items()
					if 'warped' not in k
				}
				msg = [
					f"iso: {s['sym_iso']:0.3f}",
					f"VIS: {s['sym_VIS']:0.3f}",
					f"CLA: {s['sym_CLA']:0.3f}",
					f"mi: {s['mi']:0.3f}",
					f"sym: {s['sym']:0.3f}",
					f"ovp: {s['ovp']:0.3f}",
				]
				msg = ' '.join(msg)
				msg = f"Registering {name} . . . {msg} . . . {tx_id}"
				logger.info(msg)
		df_tx = pd.DataFrame(merge_dicts(dlist))
		if not len(df_tx):
			if self.verbose:
				print(f"Warning, no good tx found for: '{name}'")
			return {}
		self._cache_txs(name, df_tx)
		best_tx_id, perf = self._bst_tx(name, df_tx, mi_q)
		best_tx = tx_all[best_tx_id]
		if save_results:
			self._sv_tx(name, best_tx, True)
			try:
				self._apply_tx_obj2lps(
					name=name,
					moving=moving_aff if
					moving_aff else moving,
					moving_orig=moving if
					moving_aff else None,
					tx_aff=tx_aff,
					tx=best_tx,
					use_mm=True,
					save_results=save_results,
				)
			except TypeError:
				msg = 'parcellation not found'
				msg = f"Registering {name} . . . {msg}"
				logger.warning(msg)

		# clean up tmp dir
		path = pathlib.Path('/tmp')
		for p in path.rglob('*.nii.gz'):
			os.remove(str(p))
		for p in path.rglob('*.mat'):
			os.remove(str(p))

		results = {
			'df': df_tx,
			'tx_id': best_tx_id,
			'moving_aff': moving_aff,
			'perf': perf,
		}
		return results

	def load_tx(
			self,
			name: str,
			mode: str = None,
			is_inv: bool = True, ):
		if mode is not None:
			self.set_mode(mode)

		# nonlinear
		tx = [
			name, f"space-{self.space}{self.dim}",
			f"ants-{'inv' if is_inv else 'fwd'}",
		]
		tx = '_'.join(tx)
		tx = list(filter(
			lambda f: tx in f,
			os.listdir(self.mice.cfg.tx_dir)
		))
		tx = [
			pjoin(self.mice.cfg.tx_dir, f)
			for f in sorted(tx, reverse=not is_inv)
		]

		# affine
		tx_file = [
			name, f"space-{self.space}{self.dim}",
			'ants-aff.mat',
		]
		tx_file = '_'.join(tx_file)
		try:
			tx_file = next(
				f for f in os.listdir(self.mice.cfg.tx_dir)
				if tx_file in f
			)
			tx_file = pjoin(self.mice.cfg.tx_dir, tx_file)
			tx_aff = ants.read_transform(tx_file)
		except StopIteration:
			tx_aff = None
		return tx, tx_aff

	def load_in_house(self, res: int = None):
		res = res if res else self.in_house_res
		spacing_mm = res / 1000
		file = [
			f"{self.in_house_name}",
			f"{spacing_mm:.2f}.nii.gz",
		]
		file = '_'.join(file)
		file = pjoin(
			self.mice.cfg.base_dir, 'Yale',
			'atlases_templates', file,
		)
		return ants.image_read(file)

	def reorient(self, img: ants.ANTsImage):
		if len(img.shape) == 3:
			tr_axes = self.tr_axes
		elif len(img.shape) == 4:
			tr_axes = self.tr_axes + (3,)
		else:
			raise ValueError(img.shape)
		x = reorient_arr(img.numpy(), tr_axes=tr_axes)
		origin = tuple(img.origin[i] for i in tr_axes)
		spacing = tuple(img.spacing[i] for i in tr_axes)
		return reset_ants_img(x, spacing, origin, img.dtype)

	def to_lps(self, img, spacing=None, dtype='uint32'):
		x = reorient_arr(img.numpy(), self.tr_axes)
		x = x[..., ::-1]
		x = x[::-1]
		origin = tuple(img.origin[i] for i in self.tr_axes)
		if spacing is None:
			spacing = tuple(img.spacing[i] for i in self.tr_axes)
		return reset_ants_img(x, spacing, origin, dtype, False)

	def _cache_txs(self, name, df_tx, tx_all=None):
		path = pjoin(self.mice.cfg.tx_dir, 'cache')
		os.makedirs(path, exist_ok=True)
		save_obj(
			obj=df_tx,
			file_name=f"perf_{name}_({now()})",
			save_dir=path,
			mode='df',
			verbose=False,
		)
		if tx_all is not None:
			path = pjoin(self.mice.cfg.tx_dir, 'cache', name)
			os.makedirs(path, exist_ok=True)
			for label, tx in tx_all.items():
				self._sv_tx(label, tx, False, path)
		return

	def _bst_tx(self, name, df_tx, mi_q):
		mi = df_tx.loc[df_tx['metric'] == 'mi']
		sym = df_tx.loc[df_tx['metric'] == 'sym']
		mi = mi.reset_index(drop=True)
		sym = sym.reset_index(drop=True)
		cond = mi['score'] > np.nanquantile(mi['score'], mi_q)
		_df = sym.loc[cond]
		try:
			best_params = _df.iloc[_df['score'].argmax()]
		except ValueError:
			cond = mi['score'] > np.nanquantile(mi['score'], 3 * mi_q / 4)
			_df = sym.loc[cond]
			best_params = _df.iloc[_df['score'].argmax()]
		best_params = best_params.drop(['metric', 'score'])
		best_params = dict(best_params)
		best_tx_id = _mk_tx_string(
			seed=best_params['seed'],
			sm=best_params['syn_metric'],
			am=best_params['aff_metric'],
			t=best_params['type'],
			g=best_params['grad'],
			syn=best_params['syn_bin'],
			aff=best_params['aff_bin'],
			r=best_params['aff_rate'],
		)
		cond = functools.reduce(
			lambda x, y: x & y, [
				df_tx[k] == v for k, v
				in best_params.items()
			],
		)
		_df = df_tx.loc[cond]
		if self.verbose:
			msg = f"{name}, perf:\n\n"
			msg += f"{_df.to_string(index=False)}\n"
			print(msg)
		perf = _df.loc[_df['metric'] == 'sym', 'score'].item()
		return best_tx_id, perf

	def _sv_tx(self, name, tx, save_warped=False, path=None):
		path = path if path else self.mice.cfg.tx_dir
		dst = [
			name,
			f"space-{self.space}{self.dim}",
			'ants',
		]
		dst = '_'.join(dst)
		for src in tx['fwdtransforms']:
			ext = '.'.join(src.split('.')[1:])
			_dst = pjoin(path, f"{dst}-fwd.{ext}")
			shutil.copyfile(src=src, dst=_dst)
		for src in tx['invtransforms']:
			ext = '.'.join(src.split('.')[1:])
			_dst = pjoin(path, f"{dst}-inv.{ext}")
			shutil.copyfile(src=src, dst=_dst)

		if save_warped:
			warped = tx.get('warpedmovout', None)
			if warped is None:
				return
			warped_file = [
				name,
				f"space-{self.space}{self.dim}",
				'T1w',
			]
			warped_file = '_'.join(warped_file)
			warped_file += '.nii.gz'
			warped_file = pjoin(
				self.mice.cfg.warped_dir, warped_file)
			ants.image_write(warped, warped_file)
		return

	def img_in_ccf(self, x, dtype: Union[str, Type] = 'uint32'):
		return reset_ants_img(
			img=x, dtype=dtype,
			origin=self.mice.al.origin,
			spacing=self.mice.al.mcc.resolution,
		)

	def _apply_tx_obj2lps(
			self,
			name: str,
			moving: ants.ANTsImage,
			moving_orig: ants.ANTsImage,
			tx: Union[dict, List[str]],
			tx_aff: ants.ANTsTransform,
			save_results: bool = True,
			extras: bool = True,
			use_mm: bool = True,
			obj: dict = None,):

		if obj is None:
			obj = [
				'cols',
				'brain',
				'cortex',
				'subcortex',
				'cortex-top',
				'brain-eroded',
			]
			if self.mode == 'bold':
				obj = {
					item: self.mice.parcel[item][:]
					for item in obj
				}
			else:
				raise NotImplementedError
		if extras:
			obj['CH'] = self.mice.al.get_masks(
				'CH', True)
			obj['major'] = self.mice.al.get_masks(
				self.mice.al.get_major_structs(), True)
		obj = {
			k: self.img_in_ccf(v)
			for k, v in obj.items()
		}
		obj = {**obj, **self.objects}
		obj = self.apply2objects(
			obj_fixed=obj,
			moving=moving,
			moving_orig=moving_orig,
			tx_aff=tx_aff,
			tx=tx,
		)
		if save_results:
			self.save_png(
				name=name,
				objects=obj,
				fixed=moving
				if moving_orig is
				None else moving_orig,
			)

		if moving_orig is not None:
			spacing = moving_orig.spacing
		else:
			spacing = moving.spacing
		if any(e > 1.0 for e in spacing) and use_mm:
			spacing = tuple(e / 1000 for e in spacing)
		obj = {
			k: self.to_lps(v, spacing)
			for k, v in obj.items()
		}
		obj['eroded'] = _erode_post(obj['brain'])
		if save_results:
			for label, img in obj.items():
				file = [
					name,
					f"space-indiv{self.dim}",
					label,
				]
				file = f"{'_'.join(file)}.nii.gz"
				file = pjoin(self.mice.cfg.masks_dir, file)
				ants.image_write(img, file)
		return obj

	def load_obj_masks(
			self,
			name: str,
			mask_names: Union[str, List[str]] = None,
	) -> Dict[str, ants.ANTsImage]:
		masks = {}
		mask_names = mask_names if mask_names else [
			'cols' if self.mode == 'ca2' else 'brain'
		]
		if not isinstance(mask_names, list):
			mask_names = [str(mask_names)]
		for mask_name in mask_names:
			file = [
				name,
				f"space-indiv{self.dim}",
				f"{mask_name}.nii.gz",
			]
			file = '_'.join(file)
			f = pjoin(self.mice.cfg.masks_dir, file)

			# load img
			img = ants.image_read(
				filename=f, pixeltype='unsigned int')
			if self.mode == 'ca2' and name != self.in_house_name:
				img = reset_ants_img(
					img=img.numpy().squeeze(),
					spacing=img.spacing[:2],
					origin=0, use_um=False,
				)
				mask = self.mice.mask2d[name].load_ants('unsigned int')
				img = ants.resample_image_to_target(
					image=img, target=mask,
					interp_type='genericLabel',
				)
				img *= mask
			masks[mask_name] = img
		return masks

	def get_accepted_regions(self):
		if self.mode == 'bold':
			regions_obj_name = 'brain'
			anat_masks = self.mice.mask3d
			regions = self.mice.parcel[regions_obj_name]
		else:
			regions_obj_name = 'cols'
			anat_masks = self.mice.mask2d
			regions = 'parcel2d-top_10um.nii.gz'
			regions = pjoin(self.mice.cfg.parcel_dir, regions)
			regions = ants.image_read(regions).numpy()

		regions = regions[regions.nonzero()]
		regions = np.unique(regions)

		df_vox = []
		for key, anat in self.mice.T1w.items():
			m = anat_masks[key].load_ants()
			m = m.numpy().astype(bool)
			obj_masks = self.load_obj_masks(key, regions_obj_name)
			obj = obj_masks[regions_obj_name].numpy() * m
			sizes = {
				r: len(v) for r, v in
				unique_idxs(obj).items()
			}
			sizes = {
				i: sizes.get(i, 0)
				for i in regions
			}
			_info = self.mice.parcel.df.loc[
				self.mice.parcel.df['id'].isin(sizes)]
			df_vox.append({
				'key': [key] * len(sizes),
				'obj': list(obj_masks) * len(sizes),
				'acro': _info['acro'].values,
				'hemis': _info['acro'].values,
				'layer': _info['acro'].values,
				'region_id': sizes.keys(),
				'size': sizes.values(),
			})
		df_vox = pd.DataFrame(merge_dicts(df_vox))
		keys = sorted(df_vox['key'].unique())

		# Remove below cutoff
		remove = []
		for key in keys:
			_df = df_vox.loc[
				(df_vox['key'] == key) &
				(df_vox['obj'] == regions_obj_name)
				]
			small = _df.loc[
				_df['size'] <= self.roi_cutoff,
				'region_id',
			].values
			small = np.array(sorted(small))
			small_sym = self.mice.parcel.complement(small)
			remove.extend(list(small_sym))
		remove = collections.Counter(remove)
		remove = {
			k: len(keys) - remove.get(k, 0)
			for k in regions
		}
		freq = {
			k: float(v) / len(keys)
			for k, v in remove.items()
		}
		freq = dict(sorted(freq.items()))
		acc = [int(i) for i in regions if freq[i] > self.roi_thres]
		rej = [int(i) for i in regions if i not in acc]

		output = {
			'df': df_vox,
			'freq': freq,
			'acc': acc,
			'rej': rej,
		}
		msg = ''
		if self.verbose:
			msg += f"Name: {regions_obj_name} . . . "
			msg += f"a total of: {len(acc)} regions "
			msg += f"accepted, {len(rej)} regions rejected.\n"
			msg += f"accept ratio: {100 * len(acc) / len(freq):0.2f} %, "
			msg += f"reject ratio: {100 * len(rej) / len(freq):0.2f} %, "
			msg += f"using a presence threshold of {self.roi_thres * 100} %\n\n"
			print(msg)
		return output

	def get_tsnr_masks(
			self,
			q_bold: float = 0.05,
			acc_bold: Iterable[int] = None, ):
		tsnr_masks_ca = {}
		tsnr_masks_bold = {}
		# CA2
		meta_dir = pjoin(self.mice.cfg.base_dir, 'meta_info')
		df = pd.read_csv(pjoin(meta_dir, 'QualityControl_CA.csv'))
		for i, row in df.iterrows():
			sub, ses, run, q_tsnr, q_mu, _, _ = row
			key = self.mice.get_key(sub, ses, run)
			good_pix = compute_tsnr_mask(
				mice=self.mice,
				key=key,
				mode='ca2',
				mask=None,
				q_tsnr=q_tsnr,
				q_mu=q_mu,
			)
			if good_pix is None:
				continue
			tsnr_masks_ca[key] = good_pix
		# BOLD
		_mode = self.mode
		self.set_mode('bold')
		if acc_bold is None:
			acc_bold = self.get_accepted_regions()
			acc_bold = acc_bold['acc']
		looper = itertools.product(
			self.mice.cfg.sub_ids,
			self.mice.cfg.ses_ids,
		)
		for sub, ses in looper:
			key = self.mice.get_key(sub, ses)
			mask = self.mice.mask3d[key].load_ants()
			if mask is None:
				continue
			else:
				mask = mask.numpy()
			f = f"{key}_space-indiv3d_brain.nii.gz"
			f = pjoin(self.mice.cfg.masks_dir, f)
			obj = ants.image_read(f).numpy().astype(int)
			obj[~np.isin(obj, acc_bold)] = 0
			obj = obj > 0
			for run in self.mice.cfg.run_ids:
				key = self.mice.get_key(sub, ses, run)
				good_vox = compute_tsnr_mask(
					mice=self.mice,
					key=key,
					mode='bold',
					mask=obj,
					q_tsnr=q_bold,
				)
				if good_vox is None:
					continue
				bad_vox = np.logical_and(obj, ~good_vox)
				tsnr_masks_bold[key] = np.logical_and(mask, ~bad_vox)
		self.set_mode(_mode)
		output = {
			'ca2': tsnr_masks_ca,
			'bold': tsnr_masks_bold,
		}
		return output

	def _organize_func(
			self,
			description: str,
			tsnr_masks: Dict[str, np.ndarray],
			accepted_regions: List[int],
			logger: logging.Logger, ):
		if self.mode == 'bold':
			mask_indiv = self.mice.mask3d
			preproc_data = self.mice.bold_preproc
			save_dir = self.mice.cfg.bold_dir
			roi_key = 'eroded'
			ndims = 3
		elif self.mode == 'ca2':
			mask_indiv = self.mice.mask2d
			preproc_data = self.mice.ca_preproc
			save_dir = self.mice.cfg.ca_dir
			roi_key = 'cols'
			ndims = 2
		else:
			raise ValueError(f"invalid mode: '{self.mode}'")

		pbar = tqdm(
			preproc_data.items(),
			total=len(preproc_data),
			dynamic_ncols=True,
			leave=self.verbose,
			position=0,
		)
		for key, func in pbar:
			fname = '_'.join([
				key,
				f"task-{func.task}",
				f"space-{self.space}{self.dim}",
				f"desc-{description}",
				func.mode,
			])
			_f = pjoin(save_dir, f"{fname}.npy")
			if os.path.isfile(_f) or func.nii_file is None:
				continue
			msg = f"Organizing {key} ({self.mode} / {description})"
			pbar.set_description(msg)
			msg += '  .  .  .  '

			x = func.load_ants().numpy()
			x = x.astype(float).squeeze()
			shape = x.shape[:ndims]

			sub, ses, run = self.mice.looper[key]
			_k = self.mice.get_key(sub, ses)
			obj = self.load_obj_masks(_k, roi_key)
			obj = obj[roi_key]
			if obj.shape != shape:
				obj = resample_to_target(
					img=obj,
					shape=shape,
					dtype='uint32',
					interp_type='genericLabel',
				)
			roi = obj.numpy().astype('uint32')
			m_indiv = mask_indiv[_k].load_ants()
			if m_indiv.shape != shape:
				m_indiv = resample_to_target(
					img=m_indiv,
					shape=shape,
					dtype='float',
					interp_type='linear',
				)
			m_indiv = m_indiv.numpy() == 1.0
			m_tsnr = tsnr_masks.get(key)
			if m_tsnr is not None:
				deleted = np.logical_and(m_indiv, ~m_tsnr).sum()
				deleted = deleted / m_indiv.sum() * 100
				msg += f"tSNR mask deleted pix/vox:  {deleted:0.1f} %"
				m_indiv = np.logical_and(m_indiv, m_tsnr)

			data_arr = []
			for ii in accepted_regions:
				m = np.logical_and(roi == ii, m_indiv)
				data_arr.append(x[m].mean(0, keepdims=True))
			data_arr = np.concatenate(data_arr)
			_ = save_obj(
				obj=data_arr,
				file_name=fname,
				save_dir=save_dir,
				verbose=False,
				mode='npy',
			)
			# add logs
			nans = np.isnan(data_arr).sum(1) == 0
			msg += f"  |  # nonan: {nans.sum()}"
			msg += f",  shape: {data_arr.shape}"
			logger.info(msg)
		return


def reorient_arr(x: np.ndarray, tr_axes: tuple):
	a, b = len(x.shape), len(tr_axes)
	assert b in [a, a - 1]
	if b == a - 1:
		tr_axes += (a - 1, )
	return np.transpose(x, tr_axes)[::-1]


def resample_to_target(
	img: ants.ANTsImage,
	shape: Iterable[int],
	dtype: str = 'uint32',
	interp_type: str = 'genericLabel', ):
	spacing = [
		i / j for i, j in
		zip(img.shape, shape)
	]
	spacing = [
		s * ratio for s, ratio in
		zip(img.spacing, spacing)
	]
	tgt = np.ones(shape).astype(dtype)
	tgt = ants.from_numpy(
		data=tgt,
		origin=img.origin,
		spacing=spacing,
		direction=img.direction,
	)
	return ants.resample_image_to_target(
		image=img,
		target=tgt,
		imagetype=0,
		interp_type=interp_type,
	)


def reset_ants_img(
		img: Union[np.ndarray, ants.ANTsImage],
		spacing: Union[float, int, tuple, list] = None,
		origin: Union[float, int, tuple, list] = None,
		dtype: Union[str, Type] = None,
		use_um: bool = True, ):

	if isinstance(img, ants.ANTsImage):
		origin = origin if origin else img.origin
		spacing = spacing if spacing else img.spacing
		img = img.numpy()
	else:
		origin = origin if origin else 0
		spacing = spacing if spacing else 1

	ndims = len(img.shape)
	if isinstance(origin, (float, int)):
		origin = [origin] * ndims
	if isinstance(spacing, (float, int)):
		spacing = [spacing] * ndims

	if any(e < 1 for e in spacing) and use_um:
		spacing = [int(e * 1000) for e in spacing]

	ants_img = ants.from_numpy(
		img.astype(dtype if dtype else img.dtype),
		origin=origin, spacing=spacing)
	return ants_img


def _erode_post(brain, radius='boundary'):
	from .parcellation import erode_repair
	eroded = erode_repair(brain.numpy(), radius)
	eroded = reset_ants_img(
		img=eroded,
		spacing=brain.spacing,
		origin=brain.origin,
		dtype=brain.dtype,
		use_um=False,
	)
	return eroded


def _mk_tx_string(seed, sm, am, t, g, syn, aff, r):
	msg = [
		f"seed-{seed}",
		f"synmetric-{sm}",
		f"affmetric-{am}",
		f"type-{t}",
		f"grad-{g:0.2f}",
		f"synbin-{syn}",
		f"affbin-{aff}",
		f"affrate-{r}",
	]
	return '_'.join(msg)


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
		"--cutoff",
		help='minimum cutoff ROI vol',
		type=int,
		default=0,
	)
	parser.add_argument(
		"--thres",
		help='acceptance threshold for ROIs',
		type=float,
		default=0.2,
	)
	parser.add_argument(
		"--seeds",
		help='number of seeds',
		type=int,
		default=10,
	)
	parser.add_argument(
		"--space",
		help='common space name',
		type=str,
		default='CCF',
	)
	parser.add_argument(
		"--anat",
		help='register anatomical data first?',
		action='store_true',
		default=False,
	)
	parser.add_argument(
		"--verbose",
		help='prints progress',
		action='store_true',
		default=False,
	)
	return parser.parse_args()


def main():
	args = _setup_args()
	from model.mouse import Mice
	cfg = (args.nn, args.ll, args.resolution)
	mice = Mice(
		cfg=cfg,
		full_load=True,
		load_parcel=True,
		load_allen=True,
	)
	register = Register(
		mice=mice,
		space=args.space,
		verbose=args.verbose,
		in_house_name='Symmetric_N162',
	)
	if args.anat:
		_ = register.register_anat(
			save_results=True,
			n_seeds=args.seeds,
		)
	register.organize_func(
		cutoff=args.cutoff,
		thres=args.thres,
	)
	print('\n')
	print(args)
	print("[PROGRESS] register.register done.\n")


if __name__ == "__main__":
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		main()
