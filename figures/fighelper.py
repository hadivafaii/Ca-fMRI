from analysis.helper import *
from matplotlib.patches import Rectangle
from analysis.bootstrap import corr_group


def ctx2ax(
		ax,
		base,
		x: np.ndarray,
		vmin: float = None,
		vmax: float = None,
		contour: bool = True,
		cbar: bool = True,
		proj: str = 'top',
		**kwargs, ):
	defaults = {
		'cmap': 'icefire',
		'cbar_shrink': 0.7,
		'cbar_aspect': 30,
		'cbar_loc': 'left',
		'cbar_tick_nbins': 4,
		'contour_width': 0.2,
		'contour_alpha': 0.8,
		'contour_coarse': False,
		'contour_cmap': 'Greys',
		'brain_cmap': 'Greys_r',
		'interp': 'antialiased',
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	tr = base.trim2d(base.project_vec(
		vec=x,
		add_disjoint=False,
	))
	mask = base.trim2d(base.project_vec(
		vec=np.isfinite(x),
		add_disjoint=False,
	))[f'proj-{proj}'][0]
	if isinstance(contour, np.ndarray):
		pass
	elif contour:
		contour = base.get_contour(
			width=kwargs['contour_width'],
			coarse=kwargs['contour_coarse'],
		)
	else:
		contour = None
	ax.imshow(
		X=mwh(tr[proj] == 0, tr[proj]),
		cmap=kwargs['brain_cmap'],
		interpolation='none',
	)
	im = ax.imshow(
		X=mwh(mask == 0, tr[f'proj-{proj}'][0]),
		vmin=vmin,
		vmax=vmax,
		cmap=kwargs['cmap'],
		interpolation=kwargs['interp'],
	)
	if cbar:
		cbar = plt.colorbar(
			ax=ax,
			mappable=im,
			location=kwargs['cbar_loc'],
			shrink=kwargs['cbar_shrink'],
			aspect=kwargs['cbar_aspect'],
		)
		cbar.ax.yaxis.set_ticks_position(
			position=kwargs['cbar_loc'],
		)
		cbar.ax.locator_params(
			axis='y',
			nbins=kwargs['cbar_tick_nbins'],
		)
	else:
		cbar = None
	if contour is not None:
		ax.imshow(
			X=mwh(~contour, contour),
			cmap=kwargs['contour_cmap'],
			alpha=kwargs['contour_alpha'],
			interpolation='none',
		)
	return ax, cbar


def pairwise_lite(
		attr: str,
		bt_dict: dict,
		good_all: np.ndarray = None,
		n_resamples: int = int(1e5),
		add_labels: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (11, 4),
		'tval': get_tval(9, 0.95),
		'palette': prepare_palette(),
		'ylim_lower': -10 if attr == 'deg' else -0.03,
		'line_lw': 5,
		'line_alpha': 0.8,
		'scatter_s': 30,
		'scatter_alpha': 0.5,
		'line_color': 'dimgrey',
		'scatter_color': 'C7',
		'scatter_edgecolor': 'w',
		'annot_xy': {
			0: (0.395, 0.92) if attr == 'deg' else (0.03, 0.92),
			1: (0.395, 0.92) if attr == 'deg' else (0.03, 0.92),
			2: (0.145, 0.92) if attr == 'deg' else (0.03, 0.92)},
		'annot_fontsize': 16,
		'label_fontsize': 17,
		'legend_fontsize': 14,
		'legend_loc': 'best',
		'tick_direction': 'in',
		'tick_labelsize': 13,
		'tick_nbins': 6,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	assert attr in ['deg', 'h', 'hr'], f"wrong attribute: '{attr}'"
	attr = f"run_{attr}"

	fig, axes = create_figure(
		nrows=1,
		ncols=3,
		sharey='row',
		figsize=kwargs['figsize'],
		constrained_layout=True,
		tight_layout=False,
	)
	for idx, (i, j) in enumerate(itertools.combinations(range(3), 2)):
		lbl_1, o_1 = list(bt_dict.items())[i]
		lbl_2, o_2 = list(bt_dict.items())[j]
		x_1 = avg(getattr(o_1, attr), 1)
		x_2 = avg(getattr(o_2, attr), 1)
		if good_all is None:
			good = np.logical_and(
				np.isfinite(x_1),
				np.isfinite(x_2),
			)
		else:
			good = good_all
		# scatter
		ax = axes.flat[idx]
		ax.scatter(
			x=x_1[good],
			y=x_2[good],
			s=kwargs['scatter_s'],
			alpha=kwargs['scatter_alpha'],
			color=kwargs['scatter_color'],
			edgecolor=kwargs['scatter_edgecolor'],
		)
		# linear regression
		lr = sp_stats.linregress(
			x=x_1[good],
			y=x_2[good],
		)
		x_min = x_1[good].min()
		x_max = x_1[good].max()
		xs = np.linspace(x_min, x_max, 1000)
		ys = lr.slope * xs + lr.intercept
		if kwargs['ylim_lower']:
			ys[ys < kwargs['ylim_lower'] * 0.7] = np.nan
		ax.plot(
			xs, ys,
			lw=kwargs['line_lw'],
			alpha=kwargs['line_alpha'],
			color=kwargs['line_color'],
		)
		if add_labels:
			ax.set_xlabel(
				xlabel=lbl_1,
				fontsize=kwargs['label_fontsize'],
			)
			if idx < 2:
				ax.set_ylabel(
					ylabel=lbl_2,
					fontsize=kwargs['label_fontsize'],
				)

		rho, se = corr_group(
			x=getattr(o_1, attr),
			y=getattr(o_2, attr),
			n_resamples=n_resamples,
		)
		text = r"$r = $" + f"{rho[0]:0.2f} " + r"$\pm$"
		text += f" {se * kwargs['tval']:0.2f}"
		ax.annotate(
			text=text,
			xy=kwargs['annot_xy'][idx],
			xycoords='axes fraction',
			fontsize=kwargs['annot_fontsize'],
		)
		ax.locator_params(
			axis='both',
			nbins=kwargs['tick_nbins'],
		)
		set_tickparams(ax, **kwargs)
	if kwargs['ylim_lower']:
		ylim = axes[-1].get_ylim()
		ylim = np.maximum(kwargs['ylim_lower'], ylim)
		axes[-1].set_ylim(ylim)
	_ = ax_square(axes)

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def pairwise(
		attr: str,
		bt_dict: dict,
		num_bins: int = 10,
		n_resamples: int = int(1e6),
		add_labels: bool = False,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (9, 9),
		'tval': get_tval(9, 0.95),
		'palette': prepare_palette(),
		'ylim_lower': -12 if attr == 'deg' else -0.03,
		'scatter_kws': {
			's': 30,
			'alpha': 0.5,
			'edgecolor': 'w'},
		'scatter_color': 'C7',
		'annot_xy': {
			(1, 0): (0.32, 0.91) if attr == 'deg' else (0.03, 0.91),
			(2, 0): (0.32, 0.91) if attr == 'deg' else (0.03, 0.91),
			(2, 1): (0.15, 0.91) if attr == 'deg' else (0.03, 0.91)},
		'annot_fontsize': 16,
		'label_fontsize': 17,
		'legend_fontsize': 14,
		'legend_loc': 'best',
		'tick_direction': 'in',
		'tick_labelsize': 13,
		'tick_nbins': 5,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	assert attr in ['deg', 'h', 'hr'], f"wrong attribute: '{attr}'"
	attr = f"run_{attr}"

	fig, axes = create_figure(
		nrows=3,
		ncols=3,
		sharex='col',
		figsize=kwargs['figsize'],
		constrained_layout=True,
		tight_layout=False,
	)
	axes[2, 0].get_shared_y_axes().join(axes[2, 0], axes[2, 1])

	for i, (mode, o) in enumerate(bt_dict.items()):
		ax = axes[i, i]
		sns.histplot(
			data=avg(getattr(o, attr), 1),
			color=kwargs['palette'][mode],
			bins=num_bins,
			stat='count',
			label=mode,
			ax=ax,
		)
		ax.legend(
			loc=kwargs['legend_loc'],
			fontsize=kwargs['legend_fontsize'],
		)
		ax.set_ylabel('')

	for j, i in itertools.combinations(range(3), 2):
		lbl_1, o_1 = list(bt_dict.items())[j]
		lbl_2, o_2 = list(bt_dict.items())[i]
		x_1 = avg(getattr(o_1, attr), 1)
		x_2 = avg(getattr(o_2, attr), 1)

		ax = axes[i, j]
		sns.regplot(
			x=x_1,
			y=x_2,
			ci=None,
			color=kwargs['scatter_color'],
			scatter_kws=kwargs['scatter_kws'],
			ax=ax,
		)
		if add_labels:
			if i == 2:
				ax.set_xlabel(
					xlabel=lbl_1,
					fontsize=kwargs['label_fontsize'],
				)
			if j == 0:
				ax.set_ylabel(
					ylabel=lbl_2,
					fontsize=kwargs['label_fontsize'],
				)

		rho, se = corr_group(
			x=getattr(o_1, attr),
			y=getattr(o_2, attr),
			n_resamples=n_resamples,
		)
		text = r"$r = $" + f"{rho[0]:0.2f} " + r"$\pm$"
		text += f" {se * kwargs['tval']:0.2f}"
		ax.annotate(
			text=text,
			xy=kwargs['annot_xy'][(i, j)],
			xycoords='axes fraction',
			fontsize=kwargs['annot_fontsize'],
		)

	ylim = axes[2, 1].get_ylim()
	ylim = np.maximum(kwargs['ylim_lower'], ylim)
	axes[2, 1].set_ylim(ylim)

	for ax in axes.flat:
		ax.tick_params(
			axis='both',
			direction=kwargs['tick_direction'],
			labelsize=kwargs['tick_labelsize'],
		)
		ax.locator_params(
			axis='both',
			nbins=kwargs['tick_nbins'],
		)

	for j, i in itertools.combinations(range(3), 2):
		_ = ax_square(axes[i, j])

	for ax in np.diag(axes):
		ax.tick_params(
			axis='x',
			direction='out',
		)
		ax.set_yticks([])
		_ = ax_square(ax)
	axes[2, 1].tick_params(axis='y', labelleft=False)

	for ax in axes[np.triu_indices(3, k=1)]:
		ax.remove()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def cut4figs(
		attr: str,
		bt_dict: dict,
		booter: Callable,
		num_bins: int = None,
		bin_edges: Tuple[float, float] = None, ):
	if attr == 'h':
		num_bins = num_bins if num_bins else 21
		bin_edges = bin_edges if bin_edges else (0, 1)
		bins = np.linspace(*bin_edges, num_bins)
		bins[-1] += 0.0001
		xs = bins[1:].copy()
		xs[0], xs[-1] = 0, 1
		attr = f"seed_{attr}"
	elif attr == 'deg':
		num_bins = num_bins if num_bins else 70
		bin_edges = bin_edges if bin_edges else (5, 350)
		bins = np.linspace(*bin_edges, num_bins)
		bins -= 2.5
		xs = (bins[:-1] + bins[1:]) / 2
		attr = f"run_{attr}"
	else:
		raise ValueError(f"wrong attribute: '{attr}'")

	cut = {}
	for mode, o in bt_dict.items():
		x = getattr(o, attr)
		shape = (*x.shape[:x.ndim - 1], -1)

		x = x.reshape((-1, 1, x.shape[-1]))
		cut[mode] = cut_data(
			data=x,
			bins=bins,
			oc_only=True,
		)[0].reshape(shape)

	se = {}
	for mode, y in cut.items():
		if attr == 'h':
			_se = booter(np.expand_dims(bn.nanmean(y, -3), -2))
		else:
			_se = booter(np.expand_dims(y, -2))
		se[mode] = _se.squeeze()

	return cut, se, bins, xs


def set_tickparams(ax, **kwargs):
	for axis in ['x', 'y']:
		direction = kwargs.get(f'tick_direction_{axis}')
		if direction is None:
			direction = kwargs.get('tick_direction')
		labelsize = kwargs.get(f'tick_labelsize_{axis}')
		if labelsize is None:
			labelsize = kwargs.get('tick_labelsize')
		ax.tick_params(
			axis=axis,
			direction=direction,
			labelsize=labelsize,
		)
	return ax


def set_labels(ax, **kwargs):
	for axis in ['x', 'y']:
		label = kwargs.get(f'label_{axis}')
		fontsize = kwargs.get(f'label_fontsize_{axis}')
		if fontsize is None:
			fontsize = kwargs.get('label_fontsize')
		if axis == 'x':
			ax.set_xlabel(
				xlabel=label,
				fontsize=fontsize,
			)
		else:
			ax.set_ylabel(
				ylabel=label,
				fontsize=fontsize,
			)
	return ax


def show_kws(num_k: int):
	if num_k == 3:
		kws = {
			'vmin': 0.2,
			'vmax': 1.0,
			'mask_thresh': 1 / 7,
			'figsize_x': 3,
			'dpi': 80,
		}
	elif num_k == 7:
		kws = {
			'vmin': 0.2,
			'vmax': 0.9,
			'mask_thresh': 1 / 7,
			'label_fontsize': 16,
		}
	elif num_k >= 20:
		kws = {
			'vmin': 0.0,
			'vmax': 0.7,
			'mask_thresh': 0.05,
			'contour_alpha': 0.7,
			'contour_width': 0.2,
			'contour_coarse': False,
			'figsize_x': 13.5,
		}
	else:
		raise ValueError(num_k)
	kws.update({
		'add_disjoint': True,
		'titles': True,
	})
	return kws


def prepare_palette(x: float = 0.6):
	from model.mouse import BOLD, CAS, CAF
	mako = sns.color_palette('mako', as_cmap=True)
	rocket = sns.color_palette('rocket', as_cmap=True)
	return {BOLD: rocket(x), CAS: mako(x), CAF: to_rgb('C8')}
