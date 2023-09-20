from utils.generic import *


def default_descs(mode: str):
	if mode == 'ca2':
		descs = [
			'preproc',
			'awake',
			'hrf',
		]
		descs += [
			f"sig{s}" for s in
			[2, 4, 8, 12, 16, 20, 24, 32, 48]
		]
	elif mode.split('-')[0] == 'bold':
		descs = [
			'raw',
			'rabies',
			'rabies-hp',
			'rabies-nocr',
			'rabies-nowmcsf',
		]
	else:
		raise ValueError(mode)
	return descs


def remove_badt_ca(
		data: np.ndarray,
		run_duration: int,
		exclude: int,
		fs: int = 10,
		n_std: float = 5.0, ):
	# gets rid of photobleaching artifacts
	ab = run_duration - exclude
	ab = slice(- ab * fs, None)
	data = data[:, ab]
	# filter out outliers (spikes)
	gs = bn.nanmean(data, 0)
	thres = gs.std() * n_std
	bad_t = np.abs(gs - gs.mean()) > thres
	bad_t = np.logical_or(bad_t, ~np.isfinite(gs))
	data = np.delete(data, bad_t, axis=1)
	return data, ab, bad_t


def process_ca(
		mice,
		key: str,
		x: np.ndarray,
		m: np.ndarray,
		filt_std: float = 5,
		filt_ab: bool = True,
		kws_fill: dict = None,
		fill_nuisance: bool = True,
		band: List[Tuple[float, float]] = None, ):
	x = x[m].squeeze()
	fs = mice.cfg.ca_fs
	if filt_ab:
		ab = mice.cfg.run_duration - mice.cfg.exclude
		ab = range(x.shape[-1] - ab * fs, x.shape[-1])
		x = x[..., ab]
	else:
		ab = None
	if filt_std is not None:
		gs = x.mean(0)
		thres = gs.std() * filt_std
		bad_t = np.abs(gs - gs.mean()) > thres
	else:
		bad_t = None
	# try to load tmask
	df = pjoin(mice.cfg.base_dir, 'meta_info')
	df = pd.read_pickle(pjoin(df, 'temporal_mask_hp.df'))
	if key in df['key'].unique():
		tmask = df.loc[
			df['key'] == key,
			'temporal_mask',
		].values.astype(bool)
		tmask[:15] = True
		tmask[-15:] = True
		tmask = np.repeat(tmask, fs)
		if ab is not None:
			tmask = tmask[ab]
		if bad_t is not None:
			tmask = np.logical_and(
				tmask, ~bad_t)
	else:
		if bad_t is not None:
			tmask = ~bad_t
		else:
			tmask = None
	# fill nuisance timepoints
	if tmask is not None and fill_nuisance:
		kws_fill = kws_fill if kws_fill else {
			'b': (0.005, fs),
			'logspace': False,
			'nf': 500,
			'dt': 1 / fs,
		}
		x = lombscargle_fill(
			x, tmask, **kws_fill)
	# bandpass
	if isinstance(band, list):
		xf = {b: butterworth(x, b, fs, 3) for b in band}
	elif isinstance(band, tuple):
		xf = {band: butterworth(x, band, fs, 3)}
	elif band is None:
		xf = None
	else:
		raise RuntimeError(f"invalid band: '{band}'")
	if xf is not None:
		tmask[:50] = False
		tmask[-50:] = False
	return x, xf, tmask, ab


def butterworth(
		x: np.ndarray,
		band: Tuple[float, float],
		fs: float,
		order: int = 3, ):
	if isinstance(band, tuple):
		assert len(band) == 2
		skip = (
			band[0] == 0 and
			band[1] == fs / 2
		)
		if skip:
			return x
		if band[1] == fs / 2:
			band = band[0]
			btype = 'high'
		elif band[0] == 0:
			band = band[1]
			btype = 'low'
		else:
			btype = 'band'
	elif isinstance(band, float):
		btype = 'high'
	else:
		raise RuntimeError
	sos = sp_sig.butter(
		N=order,
		Wn=band,
		fs=fs,
		btype=btype,
		output='sos',
	)
	x_filt = sp_sig.sosfiltfilt(
		sos=sos, x=x, axis=-1)
	return x_filt


def lombscargle_fill(
		x: np.ndarray,
		tmask: np.ndarray,
		freqs: np.ndarray = None,
		b: Tuple[float, float] = None,
		logspace: bool = True,
		nf: int = int(1e3),
		dt: float = 0.1, ):
	nt = len(tmask)
	assert x.shape[-1] == nt
	assert tmask.dtype == bool

	if freqs is None:
		b = b if b else (0.005, 1 / dt)
		if logspace:
			freqs = np.logspace(
				start=np.log10(b[0]),
				stop=np.log10(b[1]),
				num=nf,
				base=10,
			)
		else:
			freqs = np.linspace(
				start=b[0],
				stop=b[1],
				num=nf,
			)
	w = 2 * np.pi * freqs

	t = np.linspace(dt, nt * dt, nt)
	cw, sw, theta = lombscargle_mathias(t[tmask], x[:, tmask], w)
	y = lombscargle_mathias_simulate(t, w, cw, sw, theta)

	# standardize according to masked data points
	y -= y[:, tmask].mean(1, keepdims=True)
	y /= y[:, tmask].std(1, keepdims=True)
	# re-scale according to original mean/std
	y *= x[:, tmask].std(1, keepdims=True)
	y += x[:, tmask].mean(1, keepdims=True)
	# fill bad timepoints with simulated data
	y_fill = np.zeros_like(x)
	y_fill[:, tmask] = x[:, tmask]
	y_fill[:, ~tmask] = y[:, ~tmask]
	return y_fill


def lombscargle_mathias(t, x, w):
	if (w == 0).sum() > 0:
		raise ZeroDivisionError()
	if t.shape[0] != x.shape[-1]:
		raise ValueError("Input arrays do not have the same size.")
	w_t = w[:, np.newaxis].dot(t[np.newaxis, :])
	theta = (1 / (2 * w)) * np.arctan(
		np.sin(2 * w_t).sum(1) /
		np.cos(2 * w_t).sum(1)
	)
	wt = w * (
		np.expand_dims(t, 1) -
		np.expand_dims(theta, 0)
	)
	c = np.cos(wt)
	s = np.sin(wt)
	cw = x.dot(c) / (c ** 2).sum(axis=0)
	sw = x.dot(s) / (s ** 2).sum(axis=0)
	return cw, sw, theta


def lombscargle_mathias_simulate(t, w, cw, sw, theta):
	"""
	recover simulated timeseries for a given time vector t
	"""
	wt = w * (
		np.expand_dims(t, 1) -
		np.expand_dims(theta, 0)
	)
	c = np.cos(wt)
	s = np.sin(wt)
	y = c.dot(cw.T) + s.dot(sw.T)
	return y.T


def compute_tsnr_mask(
		mice,
		key: str,
		mode: str,
		mask: np.ndarray = None,
		q_tsnr: float = np.nan,
		q_mu: float = np.nan, ):
	if key is None:
		return
	results = load_tsnr(mice, key, mode)
	if results is None:
		return
	sub, ses, run = mice.looper[key]
	_key = mice.get_key(sub, ses)
	if mask is None:
		if mode == 'ca2':
			mask = mice.mask2d[_key].load_ants()
		else:
			mask = mice.mask3d[_key].load_ants()
		mask = mask.numpy() == 1.0
	mu = np.abs(results['mu'])
	tsnr = results['tSNR']

	if tsnr.shape != mask.shape:
		mask = sk_resize(mask, tsnr.shape)
	good = mask.copy()

	if ~np.isnan(q_tsnr):
		if (tsnr < 0).sum() > (tsnr > 0).sum():
			thres = np.nanquantile(tsnr[mask], 1 - q_tsnr)
			good = np.logical_and(good, tsnr < thres)
		else:
			thres = np.nanquantile(tsnr[mask], q_tsnr)
			good = np.logical_and(good, tsnr > thres)
	if ~np.isnan(q_mu):
		thres = np.nanquantile(mu[mask], 1 - q_mu)
		good = np.logical_and(good, mu < thres)
	return good


def load_tsnr(mice, key: str, mode: str):
	tsnr_dir = pjoin(mice.cfg.processed_dir, 'temporalSNR')
	fname = '_'.join([
		key,
		'space-individual',
		'desc-tsnr',
		mode,
	])
	fname = f"{fname}.npy"
	f = pjoin(tsnr_dir, fname)

	if not os.path.isfile(f):
		results = estimate_tsnr(
			mice=mice,
			key=key,
			mode=mode,
			tstart=1000 if mode == 'ca2' else 0,
		)
		if results is None:
			return
		save_obj(
			obj=results,
			file_name=fname,
			save_dir=tsnr_dir,
			verbose=False,
			mode='npy',
		)
	# noinspection PyTypeChecker
	return dict(np.load(f, allow_pickle=True).item())


def estimate_tsnr(
		mice,
		key: str,
		mode: str,
		tstart: int = 0, ):
	_, preproc, masks, _, _ = mice.get_data_containers(mode)
	func = preproc.get(key)
	if func is None or func.task is None:
		return
	x = func.load_ants().numpy().squeeze()
	m = masks[mice.get_key(*mice.looper[key][:2])]
	m = m.load_ants().numpy().astype(bool)

	mu = x[..., tstart:].mean(-1) * m
	sd = x[..., tstart:].std(-1) * m

	results = {
		'mu': mu,
		'sd': sd,
		'tSNR': mu / sd,
		'gs': x[m].mean(0),
	}
	return results
