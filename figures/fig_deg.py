from .fighelper import *


def cbar_perm_test(colors: List[str] = None):
	# noinspection PyUnresolvedReferences
	cmap = matplotlib.colors.ListedColormap(
		colors=colors if colors
		else ['b', 'w', 'r'],
		name='perm_test',
	)
	fig, cax, cbar = cbar_only(
		cmap=cmap, vertical=True, figsize=(1.3, 4))
	cax.yaxis.set_ticks([])
	cax.annotate(
		text='>',
		xy=(0.2, 0.75),
		xycoords='axes fraction',
		fontsize=60,
		color='w',
	)
	cax.annotate(
		text='=',
		xy=(0.17, 0.42),
		xycoords='axes fraction',
		fontsize=60,
		color='k',
	)
	cax.annotate(
		text='<',
		xy=(0.16, 0.08),
		xycoords='axes fraction',
		fontsize=60,
		color='w',
	)
	return fig, cax, cbar


def dist_deg(
		cut: dict,
		se: dict,
		xs: np.ndarray,
		grid: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (7, 4),
		'tval': get_tval(ci=0.95),
		'palette': prepare_palette(),
		'linewidth': 2,
		'fill_alpha': 0.5,
		'norm': None,
		'norm_nbins': 11,
		'norm_xlim_right': 0.52,
		'legend_loc': 'upper right',
		'legend_fontsize': 15,
		'tick_txt_fontsize': 14,
		'tick_txt_x': 0.43,
		'tick_txt_y': -1.5,
		'tick_direction': 'in',
		'tick_labelsize_x': 17,
		'tick_labelsize_y': 14,
		'label_fontsize': 17,
		'label_x': None,
		'label_y': None,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	fig, ax = create_figure(
		figsize=kwargs['figsize'],
		constrained_layout=True,
		tight_layout=False,
	)
	cut_avg = {
		mode: avg(y, 1) for
		mode, y in cut.items()
	}
	for mode, y_avg in cut_avg.items():
		y_low = y_avg - se[mode] * kwargs['tval']
		y_high = y_avg + se[mode] * kwargs['tval']

		ax.plot(
			xs, y_avg,
			label=mode,
			lw=kwargs['linewidth'],
			color=kwargs['palette'][mode],
		)
		ax.fill_between(
			x=xs,
			y1=y_low,
			y2=y_high,
			alpha=kwargs['fill_alpha'],
			color=kwargs['palette'][mode],
		)
	if kwargs['norm'] is not None:
		lbls = np.linspace(0, 1, kwargs['norm_nbins'])
		xticks = [e * kwargs['norm'] for e in lbls]
		ax.set(
			xticks=xticks,
			xticklabels=np.round(lbls, 2),
		)
		xlim_right = kwargs['norm']
		xlim_right *= kwargs['norm_xlim_right']
		ax.set_xlim(
			left=ax.get_xlim()[0],
			right=xlim_right,
		)
		ax.text(
			y=kwargs['tick_txt_y'],
			x=kwargs['tick_txt_x'] * kwargs['norm'],
			fontsize=kwargs['tick_txt_fontsize'],
			s=r"$\times \, $" + f"{kwargs['norm']}",
		)
	ax = set_tickparams(ax, **kwargs)
	ax = set_labels(ax, **kwargs)
	ax.legend(
		loc=kwargs['legend_loc'],
		fontsize=kwargs['legend_fontsize'],
	)
	if grid:
		ax.grid()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax
