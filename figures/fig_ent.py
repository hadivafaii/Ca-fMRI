from .fighelper import *


def dist_ent_inset(
		cut: dict,
		se: dict,
		xs: np.ndarray,
		num_k: int,
		inset_grid: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (7, 4.5),
		'tval': get_tval(ci=0.95),
		'palette': prepare_palette(),
		'lw': 1.5,
		'fill_alpha': 0.5,
		'rect_edgecolor': (0, 0, 0, 1),
		'rect_lw': 0.4,
		'inset_ylim': (0, 13),
		'inset_xlim': (0.1, 0.8),
		'inset_color': '#E6E6E6',
		'inset_alpha': 0.5,
		'inset_x0': 0.20,
		'inset_y0': 0.45,
		'inset_width': 0.79,
		'inset_height': 0.535,
		'inset_tick_labelsize': 13,
		'inset_tick_nbins': 5,
		'inset_txt_dx': {2: 0.004, 3: 0.002, 4: 0.002},
		'inset_txt_dy': {2: 0.51, 3: 0.18, 4: 0.175},
		'inset_txt_ypos': {2: 6.0, 3: 3.4, 4: 1.6},
		'inset_txt_color': 'dimgrey',
		'inset_txt_fontsize': 12,
		'inset_txt_ls': '--',
		'inset_txt_lw': 0.9,
		'legend_loc': 'upper right',
		'legend_fontsize': 15,
		'tick_txt_fontsize': 14,
		'tick_txt_x': 0.85,
		'tick_txt_y': -7.0,
		'tick_direction': 'in',
		'tick_labelsize_x': 17,
		'tick_labelsize_y': 16,
		'label_fontsize': 17,
		'tick_nbins': 7,
		'label_x': None,
		'label_y': None,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	vlines = {}
	for i in [2]:
		pi = [1 / i for _ in range(i)]
		pi += [0 for _ in range(num_k - i)]
		pi = np.array(pi)
		assert len(pi) == num_k
		assert np.equal(np.round(sum(pi), 10), 1)
		vlines[i] = sp_stats.entropy(pi) / np.log(len(pi))

	fig, ax = create_figure(
		figsize=kwargs['figsize'],
		constrained_layout=True,
		tight_layout=False,
	)
	rect = Rectangle(
		xy=(min(kwargs['inset_xlim']), min(kwargs['inset_ylim'])),
		width=max(kwargs['inset_xlim']) - min(kwargs['inset_xlim']),
		height=max(kwargs['inset_ylim']) - min(kwargs['inset_ylim']),
		facecolor=to_rgb(kwargs['inset_color']) + (kwargs['inset_alpha'],),
		edgecolor=kwargs['rect_edgecolor'],
		linewidth=kwargs['rect_lw'],
		zorder=0,
	)
	ax.add_patch(rect)

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
			lw=kwargs['lw'],
			color=kwargs['palette'][mode],
		)
		ax.fill_between(
			x=xs,
			y1=y_low,
			y2=y_high,
			alpha=kwargs['fill_alpha'],
			color=kwargs['palette'][mode],
		)
	ax = set_tickparams(ax, **kwargs)
	ax = set_labels(ax, **kwargs)
	ax.locator_params(
		axis='y',
		nbins=kwargs['tick_nbins'],
	)
	ax.text(
		x=kwargs['tick_txt_x'],
		y=kwargs['tick_txt_y'],
		fontsize=kwargs['tick_txt_fontsize'],
		s=r"$\times \, \log($" + str(num_k) + r"$)$",
	)

	# Add inset
	bounds = [
		kwargs['inset_x0'],
		kwargs['inset_y0'],
		kwargs['inset_width'],
		kwargs['inset_height'],
	]
	axins = ax.inset_axes(bounds)

	mid = np.logical_and(
		min(kwargs['inset_xlim']) <= xs,
		xs <= max(kwargs['inset_xlim']),
	)
	for mode, y_avg in cut_avg.items():
		y_low = y_avg - se[mode] * kwargs['tval']
		y_high = y_avg + se[mode] * kwargs['tval']
		axins.plot(
			xs[mid], y_avg[mid],
			label=mode,
			lw=kwargs['lw'] * 4 / 3,
			color=kwargs['palette'][mode],
		)
		axins.fill_between(
			x=xs[mid],
			y1=y_low[mid],
			y2=y_high[mid],
			alpha=kwargs['fill_alpha'],
			color=kwargs['palette'][mode],
		)
	axins.tick_params(
		axis='both',
		direction=kwargs['tick_direction'],
		labelsize=kwargs['inset_tick_labelsize'],
	)
	axins.locator_params(
		axis='y',
		nbins=kwargs['inset_tick_nbins'],
	)
	axins.set_ylim(kwargs['inset_ylim'])
	axins.legend(
		loc=kwargs['legend_loc'],
		fontsize=kwargs['legend_fontsize'],
	)
	for i, xpos in vlines.items():
		ypos = kwargs['inset_txt_ypos'][i]
		ymax = ypos / max(axins.get_ylim())
		ymax += kwargs['inset_txt_dy'][i]
		axins.axvline(
			x=xpos,
			ymin=0,
			ymax=ymax,
			ls=kwargs['inset_txt_ls'],
			lw=kwargs['inset_txt_lw'],
			color=kwargs['inset_txt_color'],
			zorder=0,
		)
		axins.text(
			x=xpos + kwargs['inset_txt_dx'][i],
			y=ypos,
			s=f"$\\log({i})$",
			color=kwargs['inset_txt_color'],
			fontsize=kwargs['inset_txt_fontsize'],
			rotation=270,
		)
	axins.patch.set_facecolor(kwargs['inset_color'])
	axins.patch.set_alpha(kwargs['inset_alpha'] * 4 / 5)
	if inset_grid:
		axins.grid()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax, axins
