from utils.generic import *
from utils.plotting import create_figure, plt
from analysis.helper import dist2sim, sp_dist


def apply_butter(
		x: np.ndarray,
		fs: float,
		band: Tuple[float, float],
		order: int = 5,
		use_zi: bool = False, ):
	if x.ndim == 1:
		x = x.reshape(1, -1)

	sos = sp_sig.butter(
		N=order,
		Wn=band,
		btype='bandpass',
		output='sos',
		fs=fs,
	)
	if use_zi:
		zi = sp_sig.sosfilt_zi(sos)
		if len(x) > 1:
			zi = np.expand_dims(zi, -1)
			zi = np.repeat(zi, len(x), axis=-1)
		x_filt, _ = sp_sig.sosfilt(
			sos=sos,
			x=x,
			zi=zi,
			axis=-1,
		)
	else:
		x_filt = sp_sig.sosfiltfilt(
			sos=sos,
			x=x,
			axis=-1,
		)
	return x_filt, sos


def bandpass_corr(
		x: np.ndarray,
		fs: float,
		bands: List[Tuple[float, float]],
		zscore: bool = False, ):

	assert x.ndim == 2, \
		'must provide a 2-dim array: rows x time'
	if not isinstance(bands, list):
		bands = [bands]

	if zscore:
		x = sp_stats.zscore(x, axis=-1)
	y, freqs = apply_fft(x, fs)

	corrs = []
	band_freqs = []
	for a, b in bands:
		band_ids = range(
			np.where(freqs >= a)[0][0],
			np.where(freqs > b)[0][0]
		)
		c = get_band_corr(y, band_ids)
		corrs.append(np.expand_dims(c, 0))
		band_freqs.append(freqs[band_ids])

	results = {
		'y': y,
		'freqs': freqs,
		'bands': bands,
		'band_freqs': band_freqs,
		'corrs': np.concatenate(corrs),
	}
	return results


def apply_fft(x: np.ndarray, fs: float):
	if x.ndim == 1:
		x = x.reshape(1, -1)
	freqs = sp_fft.rfftfreq(x.shape[1], 1 / fs)
	y = sp_fft.rfftn(x, axes=[1], norm='forward')
	return y, freqs


def get_band_corr(y: np.ndarray, band_ids=None):
	if band_ids is None:
		band_ids = range(1, y.shape[-1])
	c = dist2sim(sp_dist.squareform(sp_dist.pdist(
		X=y.real[:, band_ids], metric='cosine'
	)))
	return c


def get_band_corr_old(y: np.ndarray, band_ids=None):
	if band_ids is not None:
		y = y[:, band_ids]
	c = 2 * y.conj() @ y.T
	c = c.real
	np.fill_diagonal(c, 1.)
	return c


def plot_periodogram(
		x: np.ndarray,
		fs: float,
		kind: str = 'welch',
		figsize_x: float = 2.7,
		figsize_y: float = 3.2,
		show: bool = True, ):
	x = _prepare_input(x)
	nrows = len(x)

	if kind == 'welch':
		segment_lens = [
			256 * 2 ** i
			for i in range(6)
		]
		segment_lens = [
			e for e in segment_lens
			if e <= x.shape[-1]
		]
		ncols = len(segment_lens)

		fig, axes = create_figure(
			nrows=nrows,
			ncols=ncols,
			figsize=(figsize_x * ncols, figsize_y * nrows),
			sharey='row',
			sharex='all',
			reshape=True,
		)
		f_all = collections.defaultdict(list)
		pxx_all = collections.defaultdict(list)
		for i, _x in enumerate(x):
			for j, nperseg in enumerate(segment_lens):
				f, pxx = sp_sig.welch(_x, fs, nperseg=nperseg)
				f_all[nperseg].append(np.expand_dims(f, 0))
				pxx_all[nperseg].append(np.expand_dims(pxx, 0))
				axes[i, j].semilogy(f, pxx)
				if i == 0:
					axes[i, j].set_title(f"nperseg = {nperseg}")
				if j == 0:
					axes[i, j].set_ylabel('PSD [V**2/Hz]')
				axes[i, j].grid()
		fig.suptitle('Welch periodogram', fontsize=15, y=1.01)

		f = {
			k: np.concatenate(v) for
			k, v in f_all.items()
		}
		pxx = {
			k: np.concatenate(v) for
			k, v in pxx_all.items()
		}

	elif kind == 'simple':
		if len(x) <= 5:
			ncols = len(x)
		else:
			ncols = 5
		nrows = int(np.ceil(len(x) / ncols))
		fig, axes = create_figure(
			nrows=nrows,
			ncols=ncols,
			figsize=(figsize_x * ncols, figsize_y * nrows),
			sharey='row',
			sharex='all',
			reshape=True,
		)
		if nrows == ncols == 1:
			axes = np.reshape(axes, (1, 1))

		f_all, pxx_all = [], []
		for i in range(nrows):
			for j in range(ncols):
				idx = i * ncols + j
				f, pxx = sp_sig.periodogram(x[idx], fs)
				f_all.append(np.expand_dims(f, 0))
				pxx_all.append(np.expand_dims(pxx, 0))
				axes[i, j].semilogy(f, pxx)
				axes[i, j].set_xlabel('frequency [Hz]')
				axes[i, j].set_title(f"$x[{idx}]$")
				if i == 0:
					axes[i, j].set_ylabel('PSD [V**2/Hz]')
				axes[i, j].grid()
		fig.suptitle('Simple periodogram', fontsize=15, y=1.01)

		f = np.concatenate(f_all)
		pxx = np.concatenate(pxx_all)

	else:
		raise NotImplementedError

	if show:
		plt.show()
	else:
		plt.close()

	return (f, pxx), (fig, axes)


def plot_spectrogram(
		x: np.ndarray,
		fs: float,
		colorbar: bool = False,
		figsize_x: float = 13.5,
		figsize_y: float = 3.2,
		show: bool = True, ):
	x = _prepare_input(x)
	nrows = len(x)

	fig, axes = create_figure(
		nrows=nrows,
		ncols=2,
		figsize=(figsize_x, figsize_y * nrows),
		sharey='row',
		sharex='col',
		gridspec_kw={'width_ratios': [1, 0.27]},
	)
	f_all, t_all, sxx_all = [], [], []
	for i, _x in enumerate(x):
		f, t, sxx = sp_sig.spectrogram(_x, fs)
		f_all.append(np.expand_dims(f, 0))
		t_all.append(np.expand_dims(t, 0))
		sxx_all.append(np.expand_dims(sxx, 0))
		im = axes[i, 0].pcolormesh(
			t, f, sxx,
			shading='gouraud',
			cmap='Spectral_r',
		)
		axes[i, 0].set_ylabel('Frequency [Hz]')
		if colorbar:
			plt.colorbar(im, ax=axes[i, 0], pad=0.01)

		axes[i, 1].plot(sxx.mean(1), f)
		axes[i, 1].tick_params(
			axis='y',
			which='both',
			left=True,
			right=True,
			labelleft=True,
			labelright=True,
		)
		axes[i, 1].set_ylabel('Frequency [Hz]', rotation=-90, labelpad=20)
		axes[i, 1].yaxis.set_label_position('right')
		axes[i, 1].grid()
	axes[-1, 0].set_xlabel('Time [s]')
	axes[-1, 1].set_xlabel('Power')
	if show:
		plt.show()
	else:
		plt.close()

	f = np.concatenate(f_all)
	t = np.concatenate(t_all)
	sxx = np.concatenate(sxx_all)
	return (f, t, sxx), (fig, axes)


def _prepare_input(x):
	if x.ndim == 1:
		x = x.reshape(1, -1)
	elif x.ndim == 2:
		pass
	else:
		msg = f"x.ndim must be <= 2, got: '{x.ndim}' instead"
		raise ValueError(msg)
	return x
