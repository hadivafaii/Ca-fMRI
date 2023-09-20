from .fighelper import *


def similarity_point_all(
		scores: dict,
		std_err: dict,
		conds: Iterable[str],
		legend: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'figsize_x': 2.8,
		'figsize_y': 3.2,
		'colors': None,
		'palette': 'muted',
		'pad': 0.05,
		'delta': 1.0,
		'width': 0.2,
		'sigma': 0.01,
		'scatter_size': 20,
		'scatter_alpha': 0.8,
		'markersize': 9,
		'markers': ['o'] * 3,
		'markeredgewidth': 1.2,
		'markeredgecolor': 'k',
		'capcolor': 'k',
		'capsize': 4,
		'elinewidth': 1.2,
		'legend_fontsize': 12,
		'legend_loc': 'lower left',
		'tick_direction': 'in',
		'tick_labelsize_x': 14,
		'tick_labelsize_y': 11,
		'tick_nbins': 5,
		'ylim': (0.5, 1.0),
		'aspect': 11,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	if kwargs['colors'] is None:
		kwargs['colors'] = list(sns.color_palette(
			palette=kwargs['palette'], n_colors=3))
	assert len(kwargs['colors']) == 3

	figsize = (
		kwargs['figsize_x'] * len(list(std_err.values())[0]),
		kwargs['figsize_y'] * len(scores),
	)
	fig, axes = create_figure(
		nrows=len(scores),
		ncols=1,
		sharex='all',
		sharey='all',
		figsize=figsize,
		dpi=kwargs['dpi'],
		tight_layout=False,
		constrained_layout=True,
	)
	xticks = []
	kwargs['delta'] += 3 * kwargs['width'] + 2 * kwargs['pad']
	for row, kk in enumerate(scores):
		ax = axes[row]
		ax.grid()
		for i in range(len(std_err[kk])):
			x0 = i * kwargs['delta']
			if row == 0:
				xticks.append(x0)
			for j, k in enumerate(conds):
				y = bn.nanmean(scores[kk][:, i, j], 0)
				se = std_err[kk][i, j]
				err95 = se * get_tval(ci=0.95)
				x = x0 + (j - 1) * (kwargs['width'] + kwargs['pad'])
				ax.errorbar(
					x=x,
					y=y,
					yerr=err95,
					elinewidth=kwargs['elinewidth'],
					capthick=kwargs['elinewidth'],
					capsize=kwargs['capsize'],
					color=kwargs['capcolor'],
					zorder=2,
				)
				ax.plot(
					x, y,
					marker=kwargs['markers'][j],
					markersize=kwargs['markersize'],
					markerfacecolor=kwargs['colors'][j],
					markeredgecolor=kwargs['markeredgecolor'],
					markeredgewidth=kwargs['markeredgewidth'],
					label=_cond_label(k) if
					i == 0 else None,
					zorder=10,
					lw=0,
				)
				# indivs
				y = scores[kk][:, i, j]
				x_jit = np.array([x] * len(y))
				x_jit = add_jitter(x_jit, kwargs['sigma'])
				ax.scatter(
					x=x_jit,
					y=y,
					s=kwargs['scatter_size'],
					alpha=kwargs['scatter_alpha'],
					edgecolor=kwargs['colors'][j],
					facecolor='none',
					zorder=1,
				)
		ax.set_aspect(kwargs['aspect'])
		ax.tick_params(
			axis='y',
			direction=kwargs['tick_direction'],
			labelsize=kwargs['tick_labelsize_y'],
		)
		if legend and row == 0:
			ax.legend(
				loc=kwargs['legend_loc'],
				fontsize=kwargs['legend_fontsize'],
			)
	axes[-1].set(
		xticks=xticks,
		xticklabels=[],
		ylim=kwargs['ylim'],
	)
	axes[-1].tick_params(
		axis='x',
		direction=kwargs['tick_direction'],
		labelsize=kwargs['tick_labelsize_y'],
	)
	axes[-1].tick_params(
		axis='x',
		labelsize=kwargs['tick_labelsize_x'],
	)
	axes[-1].locator_params(
		axis='y',
		nbins=kwargs['tick_nbins'],
	)

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def similarity_point(
		results: dict,
		overall: bool,
		box: bool = False,
		legend: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'figsize': (3.2, 4.1) if overall else (7.2, 4.1),
		'layout': 'constrained',
		'colors': None,
		'palette': 'muted',
		'pad': 0.05,
		'delta': 1.0,
		'width': 0.2,
		'sigma': 0.01,
		'scatter_size': 20,
		'scatter_alpha': 0.8,
		'markersize': 9,
		'markers': None,
		'markeredgewidth': 1.2,
		'markeredgecolor': 'k',
		'capcolor': 'k',
		'capsize': 4,
		'elinewidth': 1.2,
		'legend_fontsize': 12,
		'legend_loc': 'lower left',
		'tick_direction': 'in',
		'tick_labelsize_x': 14,
		'tick_labelsize_y': 11,
		'tick_nbins': 5 if overall else 8,
		'ylim': (0.5, 1.0) if overall else (0.2, 1.0),
		'aspect': 11 if overall else 7.5,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	# prepare scores & se array
	scores, std_err = _scores_arr(results, overall)
	n = scores.shape[-1]

	# finalize some kwargs items
	if kwargs['colors'] is None:
		kwargs['colors'] = list(sns.color_palette(
			palette=kwargs['palette'], n_colors=n))
	if kwargs['markers'] is None:
		kwargs['markers'] = ['o'] * n
	assert (
		len(kwargs['colors']) ==
		len(kwargs['markers']) == n
	)

	if overall:
		conds = list(results.values())[0]
		conds = list(conds['sim_overall'])
	else:
		conds = list(results['sim'])

	fig, ax = create_figure(
		nrows=1,
		ncols=1,
		layout=kwargs['layout'],
		figsize=kwargs['figsize'],
		dpi=kwargs['dpi'],
	)
	xticks = []
	wp = kwargs['width'] + kwargs['pad']
	kwargs['delta'] += n * wp - kwargs['pad']
	# kwargs['delta'] += (
	# 	n * kwargs['width'] +
	# 	(n-1) * kwargs['pad']
	# )
	for i in range(len(std_err)):
		x0 = i * kwargs['delta']
		xticks.append(
			x0 - wp / 2 if
			n % 2 == 0
			else x0
		)
		for j, k in enumerate(conds):
			y = bn.nanmean(scores[:, i, j], 0)
			x = x0 + (j - n // 2) * wp
			ax.errorbar(
				x=x, y=y,
				yerr=std_err[i, j] * get_tval(ci=0.95),
				elinewidth=kwargs['elinewidth'],
				capthick=kwargs['elinewidth'],
				capsize=kwargs['capsize'],
				color=kwargs['capcolor'],
				zorder=2,
			)
			ax.plot(
				x, y,
				marker=kwargs['markers'][j],
				markersize=kwargs['markersize'],
				markerfacecolor=kwargs['colors'][j],
				markeredgecolor=kwargs['markeredgecolor'],
				markeredgewidth=kwargs['markeredgewidth'],
				label=_cond_label(k) if
				i == 0 else None,
				zorder=10,
				lw=0,
			)
			if box:
				err = std_err[i, j] * get_tval(ci=0.75)
				rect = Rectangle(
					xy=(x - kwargs['width'] / 2, y - err),
					width=kwargs['width'],
					height=err * 2,
					facecolor='none',
					edgecolor=kwargs['capcolor'],
					zorder=3,
				)
				ax.add_patch(rect)

			# indivs
			y = scores[:, i, j]
			x_jit = np.array([x] * len(y))
			x_jit = add_jitter(x_jit, kwargs['sigma'])
			ax.scatter(
				x=x_jit, y=y,
				s=kwargs['scatter_size'],
				alpha=kwargs['scatter_alpha'],
				edgecolor=kwargs['colors'][j],
				facecolor='none',
				zorder=1,
			)

	xticklabels = [
		f"K = {list(results)[i]}"
		if overall else f"OC-{i + 1}"
		for i in range(len(std_err))
	]
	ax.set(
		xticks=xticks,
		xticklabels=xticklabels,
		aspect=kwargs['aspect'],
		ylim=kwargs['ylim'],
	)
	ax.tick_params(
		axis='both',
		direction=kwargs['tick_direction'],
		labelsize=kwargs['tick_labelsize_y'],
	)
	ax.tick_params(
		axis='x',
		labelsize=kwargs['tick_labelsize_x'],
	)
	ax.locator_params(
		axis='y',
		nbins=kwargs['tick_nbins'],
	)
	if legend:
		ax.legend(
			loc=kwargs['legend_loc'],
			fontsize=kwargs['legend_fontsize'],
		)
	ax.grid()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def plot_ocs(
		data: dict,
		vmin: float,
		vmax: float,
		mask_thres: float,
		save_dir: str = None,
		mode2lbl: dict = None,
		kws_fig: dict = None,
		figsize_col=(3, 7),
		figsize=(3, 3.2),
		**kwargs, ):
	defaults = {
		'vmin': vmin,
		'vmax': vmax,
		'cbar': False,
		'cmap': 'magma',
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	if save_dir is not None:
		save_dir = pjoin(save_dir, 'OCs')
		os.makedirs(save_dir, exist_ok=True)
	num_k = len(list(data.values())[0])
	for i in range(num_k):
		fig, axes = create_figure(
			nrows=len(data),
			ncols=1,
			figsize=figsize_col,
			sharex='all',
			sharey='all',
			tight_layout=False,
			constrained_layout=True,
		)
		for j, (mode, pi) in enumerate(data.items()):
			x = pi[i].copy()
			x[x <= mask_thres] = np.nan
			ctx2ax(x=x, ax=axes[j], **kwargs)
		remove_ticks(axes)
		if save_dir is not None:
			fig.savefig(pjoin(save_dir, f'OC-{i + 1}.png'), **kws_fig)
		plt.close()

	kwargs_disj = kwargs.copy()
	kwargs_disj['vmin'] = 1
	kwargs_disj['vmax'] = num_k
	kwargs_disj['interp'] = 'none'
	kwargs_disj['cmap'] = get_cluster_cmap(range(num_k))
	fig, axes = create_figure(
		nrows=3,
		ncols=1,
		figsize=figsize_col,
		sharex='all',
		sharey='all',
		tight_layout=False,
		constrained_layout=True,
	)
	for j, (mode, pi) in enumerate(data.items()):
		x = np.argmax(pi, 0) + 1.0
		x[np.isnan(pi).sum(0) > 0] = np.nan
		ctx2ax(x=x, ax=axes[j], **kwargs_disj)
	remove_ticks(axes)
	if save_dir is not None:
		fig.savefig(pjoin(save_dir, f'disj.png'), **kws_fig)
	plt.close()

	for mode, pi in data.items():
		if mode2lbl and save_dir:
			lbl = mode2lbl.get(mode)
			if not lbl:
				continue
			_dir = f"{save_dir}_{lbl}"
			os.makedirs(_dir, exist_ok=True)
		else:
			_dir = None
		for i in range(num_k):
			fig, ax = create_figure(
				figsize=figsize,
				tight_layout=False,
				constrained_layout=False,
			)
			x = pi[i].copy()
			x[x < mask_thres] = np.nan
			ctx2ax(x=x, ax=ax, **kwargs)
			remove_ticks(ax)
			if _dir:
				fig.savefig(pjoin(_dir, f'OC-{i+1}.png'), **kws_fig)
			plt.close()
	return


def similarity_bar(
		results: Dict[int, dict],
		add_legend: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'figsize': (4, 3.8),
		'palette': 'muted',
		'capcolor': 'k',
		'capsize': 2.7,
		'elinewidth': 0.9,
		'legend_fontsize': 11,
		'legend_loc': 'lower left',
		'tick_direction_x': 'out',
		'tick_direction_y': 'in',
		'tick_labelsize_x': 17,
		'tick_labelsize_y': 14,
		'label_fontsize': 17,
		'label_y': None,
		'ylim': (0.5, 1.0),
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	fig, ax = create_figure(
		nrows=1,
		ncols=1,
		tight_layout=False,
		constrained_layout=True,
		figsize=kwargs['figsize'],
		dpi=kwargs['dpi'],
	)
	df = _bar_df(results)
	sns.barplot(
		data=df,
		x='k',
		y='value',
		hue='label',
		palette='muted',
		ax=ax,
	)
	for rect in ax.patches:
		x = rect.get_x() + rect.get_width() / 2
		y = rect.get_height()
		se = df.loc[df['value'] == y, 'se'].item()
		err95 = se * get_tval(ci=0.95)
		ax.errorbar(
			x=x,
			y=y,
			yerr=err95,
			capsize=kwargs['capsize'],
			capthick=kwargs['elinewidth'],
			elinewidth=kwargs['elinewidth'],
			color=kwargs['capcolor'],
		)
	ax.set(
		xlabel='',
		xticklabels=[f'K = {kk}' for kk in results],
		ylim=kwargs['ylim'],
	)
	ax = set_tickparams(ax, **kwargs)
	ax = set_labels(ax, **kwargs)
	handles, labels = ax.get_legend_handles_labels()
	ax.get_legend().remove()
	if add_legend:
		ax.legend(
			handles=handles,
			labels=labels,
			loc=kwargs['legend_loc'],
			fontsize=kwargs['legend_fontsize'],
		)
	ax.grid()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def similarity_box(
		results: Dict[int, dict],
		add_legend: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'figsize': (5, 4),
		'pad': 0.05,
		'delta': 0.3,
		'width': 0.27,
		'colors': None,
		'palette': 'muted',
		'markers': ['o'] * 3,
		'markersize': 3.1,
		'markeredgecolor': 'k',
		'markerfacecolor': 'snow',
		'capcolor': 'k',
		'capsize': 2,
		'elinewidth': 0.9,
		'legend_fontsize': 9,
		'legend_loc': 'lower left',
		'tick_direction': 'out',
		'tick_labelsize_x': 11,
		'tick_labelsize_y': 9,
		'tick_nbins': 4,
		'ylim': (0.5, 1.0),
		'aspect': 7,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	if kwargs['colors'] is None:
		kwargs['colors'] = list(sns.color_palette(
			palette=kwargs['palette'], n_colors=3))
	assert len(kwargs['colors']) == 3

	fig, ax = create_figure(
		nrows=1,
		ncols=1,
		tight_layout=False,
		constrained_layout=True,
		figsize=kwargs['figsize'],
		dpi=kwargs['dpi'],
	)
	xticks = []
	kwargs['delta'] += 3 * kwargs['width'] + 2 * kwargs['pad']
	for i, (kk, res) in enumerate(results.items()):
		x0 = i * kwargs['delta']
		xticks.append(x0)
		for j, (k, y) in enumerate(res['sim_overall'].items()):
			se = res['sim_overall_b'][k]
			se = np.nanstd(se, axis=0, ddof=1)
			err95 = se * get_tval(ci=0.95)
			err75 = se * get_tval(ci=0.75)
			x = x0 + (j - 1) * (kwargs['width'] + kwargs['pad'])
			plt.errorbar(
				x=x,
				y=y,
				yerr=err95,
				elinewidth=kwargs['elinewidth'],
				capthick=kwargs['elinewidth'],
				capsize=kwargs['capsize'],
				color=kwargs['capcolor'],
			)
			ax.plot(
				x, y,
				marker=kwargs['markers'][j],
				markersize=kwargs['markersize'],
				markeredgecolor=kwargs['markeredgecolor'],
				markerfacecolor=kwargs['markerfacecolor'],
			)
			rect = Rectangle(
				xy=(x - kwargs['width'] / 2, y - err75),
				width=kwargs['width'],
				height=err75 * 2,
				facecolor=kwargs['colors'][j],
				edgecolor=kwargs['capcolor'],
				label=_cond_label(k) if
				i == 0 else None,
				zorder=2,
			)
			ax.add_patch(rect)

	ax.set(
		aspect=kwargs['aspect'],
		ylim=kwargs['ylim'],
		xticks=xticks,
		xticklabels=[f"K$={kk}$" for kk in results],
	)
	ax.tick_params(
		axis='both',
		direction=kwargs['tick_direction'],
		labelsize=kwargs['tick_labelsize_y'],
	)
	ax.tick_params(
		axis='x',
		labelsize=kwargs['tick_labelsize_x'],
	)
	ax.locator_params(
		axis='y',
		nbins=kwargs['tick_nbins'],
	)
	if add_legend:
		ax.legend(
			loc=kwargs['legend_loc'],
			fontsize=kwargs['legend_fontsize'],
		)
	ax.grid()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def similarity_line(
		sim_dict: dict,
		sim_boot: dict,
		add_legend: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (7.2, 4.1),
		'colors': None,
		'palette': 'muted',
		'markers': ['o'] * 3,
		'markersize': 8,
		'linewidth': 1.8,
		'fill_alpha': 0.25,
		'legend_loc': 'best',
		'legend_fontsize': 17,
		'tick_direction': 'in',
		'tick_labelsize_x': 17,
		'tick_labelsize_y': 13.6,
		'label_fontsize': 17,
		'label_y': None,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	if kwargs['colors'] is None:
		kwargs['colors'] = list(sns.color_palette(
			palette=kwargs['palette'], n_colors=3))
	assert len(kwargs['colors']) == 3

	fig, ax = create_figure(
		nrows=1,
		ncols=1,
		tight_layout=False,
		constrained_layout=True,
		figsize=kwargs['figsize'],
	)
	for i, k in enumerate(sim_dict):
		y = sim_dict[k].mean(0).diagonal()
		se = np.nanstd(sim_boot[k], 0, ddof=1)
		se = se.diagonal()

		ax.plot(
			range(len(y)), y,
			lw=kwargs['linewidth'],
			color=kwargs['colors'][i],
			marker=kwargs['markers'][i],
			markersize=kwargs['markersize'],
			label=_cond_label(k),
		)
		ax.fill_between(
			x=range(len(y)),
			y1=y - se * get_tval(),
			y2=y + se * get_tval(),
			color=kwargs['colors'][i],
			alpha=kwargs['fill_alpha'],
		)
	num_k = list(sim_dict.values())[0].shape[1]
	ax.set(
		xticks=range(num_k),
		xticklabels=_ticklabels(num_k),
	)
	ax = set_tickparams(ax, **kwargs)
	ax = set_labels(ax, **kwargs)
	if add_legend:
		ax.legend(
			loc=kwargs['legend_loc'],
			fontsize=kwargs['legend_fontsize'],
		)
	ax.grid()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def similarity_heatmap(
		sim: dict,
		sim_overall: dict = None,
		add_title: bool = False,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (12, 4.6),
		'fmt': '.2f',
		'square': True,
		'linewidths': 0.1,
		'linecolor': 'dimgrey',
		'vmin': 0.0,
		'vmax': 1.0,
		'cbar': False,
		'cmap': bwor_cmap(),
		'cbar_kws': {
			'shrink': 1.0,
			'aspect': 17,
			'pad': 0.08},
		'annot': True,
		'annot_kws': {
			'fontsize': 13,
			'fontname': 'Loma'},
		'tick_pre': 'OC',
		'tick_direction': 'out',
		'tick_labelsize': 11,
		'label_fontsize': 17,
		'label_pad': None,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	fig, axes = create_figure(
		nrows=1,
		ncols=3,
		sharex='all',
		sharey='all',
		layout='constrained',
		figsize=kwargs['figsize'],
	)
	for i, ((lbl_1, lbl_2), x) in enumerate(sim.items()):
		ax = axes[i]
		if x.ndim == 3:
			s = x.mean(0)
		else:
			s = x
		sns.heatmap(s, ax=ax, **filter_kwargs(sns.heatmap, kwargs))
		ax.set_ylabel(
			ylabel=lbl_1,
			fontsize=kwargs['label_fontsize'],
			labelpad=kwargs['label_pad']
			if i == 0 else None,
		)
		ax.set_xlabel(
			xlabel=lbl_2,
			fontsize=kwargs['label_fontsize'],
			labelpad=kwargs['label_pad'],
		)
		if add_title and sim_overall:
			title = f"diag avg: {s.diagonal().mean():0.2f};  "
			title += f"overal: {sim_overall[(lbl_1, lbl_2)].mean():0.2f}"
			ax.set_title(title, fontsize=15)

		num_k = list(sim.values())[0].shape[1]
		ax.set(
			xticks=[i + 0.5 for i in range(num_k)],
			yticks=[i + 0.5 for i in range(num_k)],
			xticklabels=_ticklabels(num_k, kwargs['tick_pre']),
			yticklabels=_ticklabels(num_k, kwargs['tick_pre']),
		)
		ax.tick_params(
			axis='both',
			direction=kwargs['tick_direction'],
			labelsize=kwargs['tick_labelsize'],
		)

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def bwor_cmap(mode: str = 'cos'):
	orrd = sns.color_palette('OrRd', as_cmap=True)
	blue = make_cmap([(1, 1, 1), '#18b4fb'], show=False)
	if mode == 'cos':
		bb = [0.8, 0.6, 0.4, 0.2, 0.1]
		bb += [0.045] * 3 + [0.02, 0.01]
		bb = [blue(f) for f in bb]
		rr = [0.0, 0.1, 0.25, 0.45, 0.55, 0.63, 0.70, 0.75, 0.78]
		rr = [orrd(f) for f in rr]
	elif mode == 'corr':
		bb = ['#0070da', blue(0.80), blue(0.15)]
		rr = [orrd(0.10), orrd(0.50), '#ea0c23']
	else:
		raise ValueError(mode)
	cmap = bb + ['#fffdfb'] + rr
	cmap = make_cmap(cmap, show=False)
	return cmap


def _scores_arr(results: dict, overall: bool):
	scores, std_err = [], []
	if overall:
		axis = -2
		for res in results.values():
			_scores, _std_err = [], []
			for k, v in res['sim_overall'].items():
				_scores.append(np.expand_dims(
					v, axis=-1))
				_std_err.append(np.expand_dims(bn.nanstd(
					res['sim_overall_b'][k], 0, ddof=1), axis=-1))
			_scores = np.concatenate(_scores, axis=-1)
			_std_err = np.concatenate(_std_err, axis=-1)
			scores.append(np.expand_dims(_scores, axis=axis))
			std_err.append(np.expand_dims(_std_err, axis=axis))
	else:
		axis = -1
		for k, v in results['sim'].items():
			s = v.diagonal(axis1=-2, axis2=-1)
			scores.append(np.expand_dims(
				s, axis=axis))
			se = bn.nanstd(results['sim_b'][k], 0, ddof=1)
			std_err.append(np.expand_dims(
				se.diagonal(), axis=axis))
	scores = np.concatenate(scores, axis=axis)
	std_err = np.concatenate(std_err, axis=axis)
	return scores, std_err


def _bar_df(results: dict):
	dlist = []
	for num_k, res in results.items():
		for k, v in res['sim_overall'].items():
			se = np.nanstd(res['sim_overall_b'][k], 0, ddof=1)
			dlist.append({
				'k': [num_k],
				'label': [_cond_label(k)],
				'value': [v],
				'se': [se],
			})
	df = pd.DataFrame(merge_dicts(dlist))
	return df


def _ticklabels(num_k: int, tick_pre: str = 'OC'):
	tick_labels = [
		f"{tick_pre}{k}" for k
		in range(1, num_k + 1)
	]
	return tick_labels


def _cond_label(k):
	return f'{k[0]} vs. {k[1]}'
