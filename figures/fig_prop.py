from .fighelper import *


def proportions_line(
		df: pd.DataFrame,
		bins: np.ndarray,
		vertical: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'figsize_x': 1.9 if vertical else 1.7,
		'figsize_y': 1.8 if vertical else 1.65,
		'colors': None,
		'sigma': 0.02,
		'scatter_size': 12,
		'scatter_alpha': 0.8,
		'marker': 'o',
		'markersize': 5,
		'fill_alpha': 0.3,
		'vspan_alpha': 0.05,
		'vspan_color': '#2d3938',
		'tick_direction': 'out',
		'tick_labelsize_x': 8,
		'tick_labelsize_y': 8,
		'label_fontsize': 11,
		'label_x': None,
		'label_y': None,
		'ylim': (0, 80),
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	ocs = df['oc_idx'].unique()
	modes = df['mode'].unique()

	if kwargs['colors'] is None:
		_, colors = get_cluster_cmap(
			color_ids=range(len(ocs)),
			return_colors=True,
		)
		kwargs['colors'] = dict(zip(ocs, colors))
	assert list(kwargs['colors']) == list(ocs)

	nrows = len(ocs) if vertical else len(modes)
	ncols = len(modes) if vertical else len(ocs)
	figsize = (
		kwargs['figsize_x'] * ncols,
		kwargs['figsize_y'] * nrows,
	)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=figsize,
		sharex='all',
		sharey='all',
		dpi=kwargs['dpi'],
		tight_layout=False,
		constrained_layout=True,
	)
	for i, idx in enumerate(ocs):
		for j, mode in enumerate(modes):
			ax = axes[i, j] if vertical else axes[j, i]
			_prop(df, ax, mode, idx, bins, **kwargs)

	for i in range(nrows):
		ax = axes[i, 0]
		ax.tick_params(
			axis='y',
			labelsize=kwargs[f'tick_labelsize_y'],
		)
		if kwargs['label_y']:
			ax.set_ylabel(
				ylabel=kwargs['label_y'],
				fontsize=kwargs['label_fontsize'],
			)
	for j in range(ncols):
		ax = axes[-1, j]
		ax.tick_params(
			axis='x',
			labelsize=kwargs['tick_labelsize_x'],
		)
		ax.set_xticks(bins)
		if kwargs['label_x']:
			ax.set_xlabel(
				xlabel=kwargs['label_x'],
				fontsize=kwargs['label_fontsize'],
			)
	axes[0, 0].set_xlim((min(bins), max(bins)))
	axes[0, 0].set_ylim(kwargs['ylim'])

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def _prop(df, ax, mode, idx, bins, **kwargs):
	df_ = df.loc[
		(df['mode'] == mode) &
		(df['oc_idx'] == idx)
		]
	# scatter (subjects)
	_df = df_.loc[df_['subject'] != 'group']
	for x in _df['x'].unique():
		y = _df.loc[_df['x'] == x, 'y'].values
		x_jit = np.array([x] * len(y))
		x_jit = add_jitter(x_jit, kwargs['sigma'])
		ax.scatter(
			x=x_jit,
			y=y,
			s=kwargs['scatter_size'],
			alpha=kwargs['scatter_alpha'],
			edgecolor=kwargs['colors'][idx],
			facecolor='none',
		)

	# line (group)
	_df = df_.loc[df_['subject'] == 'group']
	ax.plot(
		_df['x'],
		_df['y'],
		marker=kwargs['marker'],
		markersize=kwargs['markersize'],
		color=kwargs['colors'][idx],
	)
	ax.fill_between(
		x=_df['x'],
		y1=_df['y'] - _df['se'] * get_tval(),
		y2=_df['y'] + _df['se'] * get_tval(),
		color=kwargs['colors'][idx],
		alpha=kwargs['fill_alpha'],
	)
	# last bin vspan
	ax.axvspan(
		xmin=bins[-2],
		xmax=bins[-1],
		color=kwargs['vspan_color'],
		alpha=kwargs['vspan_alpha'],
		zorder=1,
	)
	ax.tick_params(
		axis='both',
		direction=kwargs['tick_direction'],
	)
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.grid()
	return ax


def proportions_box(
		df_cas: pd.DataFrame,
		df_bold: pd.DataFrame,
		inset_ca: Union[dict, np.ndarray] = None,
		inset_bold: Union[dict, np.ndarray] = None,
		inset_kind: str = 'imshow',
		save_file: str = None,
		ci: float = 0.95,
		props_box: dict = None,
		props_mean: dict = None,
		props_swarm: dict = None,
		props_inset: dict = None,
		kwargs_inset: dict = None,
		display: bool = False,
		**kwargs, ):
	defaults = {
		'horizontal': True,
		'cmap': None,
		'legend': False,
		'figsize_x': 13.5,
		'figsize_y': 2 * 2.7,
		'line_scale': 1.2,
		'IQR': 0.50,
		'estimator': np.nanmean,
		'xtick_decimals': 1,
		'xtick_rotation': 0,
		'xtick_labelsize': 13,
		'xlabel_fontsize': 15,
		'ylabel_fontsize': 15,
		'ytick_labelsize': 13,
		'title_fontsize': 15,
		'title_y': 1.03,
		'ylim_base': 10,
		'ylim': None,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	if kwargs['ylim'] is None:
		upper = max(
			df_cas['mean'].max(skipna=True),
			df_bold['mean'].max(skipna=True),
		)
		upper = np.ceil(upper / kwargs['ylim_base'])
		upper *= kwargs['ylim_base']
		kwargs['ylim'] = (0, upper)

	props_box = props_box if props_box else {
		'boxprops': {
			'edgecolor': 'k',
			'facecolor': 'none',
			'lw': 1.5, 'ls': '-'},
		'medianprops': {'color': 'none', 'lw': 0},
		'whiskerprops': {'color': 'k', 'lw': 1.5},
		'capprops': {'color': 'k', 'lw': 1.5, 'zorder': 3},
		'flierprops': {'alpha': 0.0, 'zorder': 1},
		'showfliers': False,
		'showmeans': True,
		'width': 0.75,
	}
	props_mean = props_mean if props_mean else {
		'marker': 'o',
		'markerfacecolor': 'white',
		'markeredgecolor': 'k',
		'markersize': 7,
		'alpha': 1.0,
	}
	props_swarm = props_swarm if props_swarm else {
		'size': 4,
		'alpha': 0.7,
		'linewidth': 0.4,
		'zorder': 0,
	}
	props_inset = props_inset if props_inset else {
		'x0': 0.37 if inset_kind == 'kde' else 0.18,
		'y0': 0.75 if inset_kind == 'kde' else 0.53,
		'w': 0.25 if inset_kind == 'kde' else 0.50,
		'h': 0.20 if inset_kind == 'kde' else 0.42,
		'facealpha': 0.1,
		'facecolor': '#2d3938',
		'xlabel_fontsize': 9,
		'xlabel': False,
		'aspect': None,
	}
	if kwargs_inset is None:
		if inset_kind == 'kde':
			kwargs_inset = {
				'lw': 2,
				'alpha': 0.7,
				'shade': True,
				'color': None,
			}
		elif inset_kind == 'imshow':
			kwargs_inset = {
				'vmin': 0.15,
				'vmax': 0.75,
				'cmap': 'magma',
			}
		else:
			raise NotImplementedError

	k = len(df_cas['oc_idx'].unique())
	intervals = df_cas['interval'].unique()
	bins = []
	for intvl in intervals:
		bins.append(intvl.left)
		bins.append(intvl.right)
	bins = np.unique(bins)
	xticklabels = [
		np.round(x, kwargs['xtick_decimals'])
		for x in bins
	]
	xticks = np.arange(-0.5, len(xticklabels) - 0.5)
	# pal = sns.color_palette(kwargs['cmap'], n_colors=len(bins))

	if kwargs['horizontal']:
		nrows, ncols = 2, k
		figsize = (kwargs['figsize_x'], kwargs['figsize_y'])
	else:
		nrows, ncols = k, 2
		figsize = (kwargs['figsize_y'], kwargs['figsize_x'])

	whis = np.round((1 + ci) / 2, 5)
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=figsize,
		sharex='all',
		sharey='all',
		style='ticks',
		reshape=True,
	)
	axes_inset = np.empty_like(axes)
	bounds_inset = [
		props_inset['x0'],
		props_inset['y0'],
		props_inset['w'],
		props_inset['h'],
	]
	looper = zip(
		[df_bold, df_cas],
		[inset_bold, inset_ca],
		axes if kwargs['horizontal'] else axes.T,
	)
	for mod_idx, (df, inset_data, _axes) in enumerate(looper):
		for oc_idx, ax in enumerate(_axes.ravel(), start=1):
			cmap = kwargs['cmap'] if kwargs['cmap'] \
				else get_cluster_cmap(oc_idx - 1)
			pal = cmap2pal(cmap, n_colors=len(intervals), start=bins[1])
			_df = df.loc[df['oc_idx'] == oc_idx]
			color = _df['color'].unique().item()
			props_mean['markerfacecolor'] = color

			sns.swarmplot(
				x='interval',
				y='mean',
				data=_df,
				palette=pal,
				size=props_swarm['size'],
				alpha=props_swarm['alpha'],
				linewidth=props_swarm['linewidth'],
				zorder=props_swarm['zorder'],
				ax=ax,
			)
			sns.boxplot(
				x='interval',
				y='mean',
				data=_df,
				whis=np.inf,
				meanprops=props_mean,
				ax=ax,
				**props_box,
			)

			# modify box
			qs = []
			for intvl in intervals:
				df_ = _df.loc[_df['interval'] == intvl]
				dof = np.sum(~np.isnan(df_['mean'].values)) - 1
				mu = kwargs['estimator'](df_['mean'].values)
				se = df_['se'].unique().item()
				# quartiles
				t = sp_stats.t.ppf((1 + kwargs['IQR']) / 2, dof)
				q1 = mu - se * t
				q3 = mu + se * t
				# whiskers
				t = sp_stats.t.ppf(whis, dof)
				low = mu - se * t
				high = mu + se * t
				qs.append((low, q1, q3, high))
			ax = modify_box(ax, qs, axis='y')

			# line
			sns.pointplot(
				x='interval',
				y='mean',
				data=_df,
				color=color,
				markers='',
				scale=kwargs['line_scale'],
				ci=None,
				ax=ax,
			)
			if kwargs['legend']:
				ax.legend(
					[Line2D([0], [0], color=color, lw=5)],
					[f"OC_{oc_idx}"], loc='upper left',
				)

			# vspan
			ax.axvspan(
				xmin=2.5,
				xmax=3.5,
				color=props_inset['facecolor'],
				alpha=props_inset['facealpha'],
				zorder=1,
			)

			# x and y label
			if kwargs['horizontal']:
				row_i, col_i = mod_idx, oc_idx - 1
			else:
				row_i, col_i = oc_idx - 1, mod_idx
			if row_i + 1 == nrows:
				ax.set_xlabel(
					xlabel='Membership',
					fontsize=kwargs['xlabel_fontsize'],
				)
			else:
				ax.set_xlabel('')
			if col_i % ncols == 0:
				ax.set_ylabel(
					ylabel='Proportion [%]',
					fontsize=kwargs['ylabel_fontsize'],
				)
			else:
				ax.set_ylabel('')

			# ticks
			ax.set(
				xticks=xticks,
				xticklabels=xticklabels,
				ylim=kwargs['ylim'],
			)
			ax.tick_params(
				axis='y',
				labelsize=kwargs['ytick_labelsize'],
			)
			ax.tick_params(
				axis='x',
				rotation=kwargs['xtick_rotation'],
				labelsize=kwargs['xtick_labelsize'],
			)
			ax.grid(ls=':')

			# insets
			if inset_data is None:
				continue

			if inset_kind == 'kde':
				kwargs_inset['color'] = color
				facecolor = to_rgb(props_inset['facecolor'])
				facecolor += (props_inset['facealpha'],)
				x = inset_data[oc_idx - 1]
			elif inset_kind == 'imshow':
				facecolor = (0, 0, 0, 0)
				x = (
					inset_data['top'],
					inset_data['proj-top'][oc_idx - 1],
				)
			else:
				raise NotImplementedError

			_ax = add_ax_inset(
				ax=ax,
				data=x,
				kind=inset_kind,
				bounds=bounds_inset,
				aspect_eq=False,
				**kwargs_inset,
			)
			if props_inset['aspect'] is not None:
				_ax.set(
					facecolor=facecolor,
					aspect=props_inset['aspect'],
				)
			else:
				_ax.set(facecolor=facecolor)

			if inset_kind == 'kde':
				_ax.tick_params(axis='y', labelleft=False)
				if props_inset['xlabel']:
					_ax.set_xlabel(
						xlabel='Proportion [%]',
						fontsize=props_inset['xlabel_fontsize'],
					)
				_ax.set_ylabel('')
			elif inset_kind == 'imshow':
				remove_ticks(_ax)
			axes_inset[row_i, col_i] = _ax
	if display:
		plt.show()
	else:
		plt.close()
	if save_file:
		print(save_file)
		print('done.')
	return fig, axes, axes_inset


def proportions_single_box(
		df: pd.DataFrame,
		save_file: str = None,
		add_suptitle: bool = False,
		inset_data: np.ndarray = None,
		inset_kind: str = 'kde',
		props_box: dict = None,
		props_mean: dict = None,
		props_inset: dict = None,
		props_swarm: dict = None,
		**kwargs, ):
	defaults = {
		'nrows': 2,
		'cmap': None,
		'legend': False,
		'figsize_x': 13.5,
		'figsize_y': 2 * 5.0,
		'line_scale': 1.5,
		'sup_text': None,
		'sup_fontsize': 17,
		'sup_y': 1.05,
		'IQR': 0.50,
		'ci': [0.95, 0.99, 0.999],
		'estimator': np.nanmean,
		'xtick_decimals': 1,
		'xtick_rotation': 0,
		'xtick_labelsize': 13,
		'xlabel_fontsize': 15,
		'ylabel_fontsize': 15,
		'ytick_labelsize': 13,
		'title_fontsize': 15,
		'title_y': 1.03,
		'ylim_base': 10,
		'ylim': None,
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	if kwargs['ylim'] is None:
		upper = df['mean'].max()
		upper = np.ceil(upper / kwargs['ylim_base'])
		upper *= kwargs['ylim_base']
		kwargs['ylim'] = (0, upper)

	props_box = props_box if props_box else {
		'boxprops': {
			'edgecolor': 'k',
			'facecolor': 'none',
			'lw': 1.5, 'ls': '-'},
		'medianprops': {'color': 'none', 'lw': 0},
		'whiskerprops': {'color': 'k', 'lw': 1.5},
		'capprops': {'color': 'k', 'lw': 1.5, 'zorder': 3},
		'flierprops': {'alpha': 0.0, 'zorder': 1},
		'showfliers': False,
		'showmeans': True,
		'width': 0.75,
	}
	props_mean = props_mean if props_mean else {
		'marker': 'o',
		'markerfacecolor': 'white',
		'markeredgecolor': 'k',
		'markersize': 11,
		'alpha': 1.0,
	}
	props_inset = props_inset if props_inset else {
		'x0': 0.37,
		'y0': 0.75,
		'w': 0.25,
		'h': 0.20,
		'facealpha': 0.1,
		'facecolor': '#2d3938',
		'xlabel_fontsize': 9,
		'xlabel': False,
		'aspect': None,
		'kws': {'lw': 2, 'alpha': 0.7, 'shade': True},
	}
	props_swarm = props_swarm if props_swarm else {
		'size': 5,
		'alpha': 0.7,
		'linewidth': 0.4,
		'zorder': 0,
	}

	k = len(df['oc_idx'].unique())
	intervals = df['interval'].unique()
	bins = []
	for intvl in intervals:
		bins.append(intvl.left)
		bins.append(intvl.right)
	bins = np.unique(bins)
	xticklabels = [
		np.round(x, kwargs['xtick_decimals'])
		for x in bins
	]
	xticks = np.arange(-0.5, len(xticklabels) - 0.5)
	# pal = sns.color_palette(kwargs['cmap'], n_colors=len(bins))

	figs, sups, axes, axes_inset = [], [], [], []
	ncols = int(np.ceil(k / kwargs['nrows']))
	for ci in kwargs['ci']:
		whis = np.round((1 + ci) / 2, 5)
		fig, _axes = create_figure(
			nrows=kwargs['nrows'],
			ncols=ncols,
			figsize=(kwargs['figsize_x'], kwargs['figsize_y']),
			style='ticks',
			sharex='none',
			sharey='all',
			reshape=True,
		)
		_axes = trim_axs(_axes, k)
		_axes_inset = np.empty_like(_axes)
		bounds_inset = [
			props_inset['x0'],
			props_inset['y0'],
			props_inset['w'],
			props_inset['h'],
		]

		for oc_idx, ax in enumerate(_axes.ravel(), start=1):
			cmap = kwargs['cmap'] if kwargs['cmap'] \
				else get_cluster_cmap(oc_idx - 1)
			pal = cmap2pal(cmap, n_colors=len(intervals), start=bins[1])
			_df = df.loc[df['oc_idx'] == oc_idx]
			color = _df['color'].unique().item()
			props_mean['markerfacecolor'] = color

			sns.swarmplot(
				x='interval',
				y='mean',
				data=_df,
				palette=pal,
				size=props_swarm['size'],
				alpha=props_swarm['alpha'],
				linewidth=props_swarm['linewidth'],
				zorder=props_swarm['zorder'],
				ax=ax,
			)
			sns.boxplot(
				x='interval',
				y='mean',
				data=_df,
				whis=np.inf,
				meanprops=props_mean,
				ax=ax,
				**props_box,
			)

			# modify box
			qs = []
			for intvl in intervals:
				df_ = _df.loc[_df['interval'] == intvl]
				dof = np.sum(~np.isnan(df_['mean'].values)) - 1
				mu = kwargs['estimator'](df_['mean'].values)
				se = df_['se'].unique().item()
				# quartiles
				t = sp_stats.t.ppf((1 + kwargs['IQR']) / 2, dof)
				q1 = mu - se * t
				q3 = mu + se * t
				# whiskers
				t = sp_stats.t.ppf(whis, dof)
				low = mu - se * t
				high = mu + se * t
				qs.append((low, q1, q3, high))
			ax = modify_box(ax, qs, axis='y')

			# line
			sns.pointplot(
				x='interval',
				y='mean',
				data=_df,
				color=color,
				markers='',
				scale=kwargs['line_scale'],
				ci=None,
				ax=ax,
			)
			if kwargs['legend']:
				ax.legend(
					[Line2D([0], [0], color=color, lw=5)],
					[f"OC_{oc_idx}"], loc='upper left',
				)

			# vspan
			ax.axvspan(
				xmin=2.5,
				xmax=3.5,
				color=props_inset['facecolor'],
				alpha=props_inset['facealpha'],
				zorder=1,
			)

			# insets
			if inset_kind == 'kde':
				inset_kws = {
					'color': color,
					**props_inset['kws'],
				}
			elif inset_kind == 'imshow':
				raise NotImplementedError
			else:
				raise NotImplementedError

			if inset_data is not None:
				_ax = add_ax_inset(
					ax=ax,
					kind=inset_kind,
					bounds=bounds_inset,
					data=inset_data[oc_idx - 1],
					aspect_eq=False,
					**inset_kws,
				)
				facecolor = to_rgb(props_inset['facecolor'])
				facecolor += (props_inset['facealpha'],)
				if props_inset['aspect'] is not None:
					_ax.set(
						aspect=props_inset['aspect'],
						facecolor=facecolor,
					)
				else:
					_ax.set(facecolor=facecolor)

				if inset_kind == 'kde':
					_ax.tick_params(axis='y', labelleft=False)
					if props_inset['xlabel']:
						_ax.set_xlabel(
							xlabel='Proportion [%]',
							fontsize=props_inset['xlabel_fontsize'],
						)
					_ax.set_ylabel('')
				_axes_inset.ravel()[oc_idx - 1] = _ax

				if inset_kind == 'imshow':
					remove_ticks(_axes_inset)

			if (oc_idx - 1) // ncols == kwargs['nrows'] - 1:
				ax.set_xlabel(
					xlabel='Membership',
					fontsize=kwargs['xlabel_fontsize'],
				)
			else:
				ax.set_xlabel('')
			if (oc_idx - 1) % ncols == 0:
				ax.set_ylabel(
					ylabel='Proportion [%]',
					fontsize=kwargs['ylabel_fontsize'],
				)
			else:
				ax.set_ylabel('')

			# ticks
			ax.set(
				xticks=xticks,
				xticklabels=xticklabels,
				ylim=kwargs['ylim'],
			)
			ax.tick_params(
				axis='y',
				labelsize=kwargs['ytick_labelsize'],
			)
			ax.tick_params(
				axis='x',
				rotation=kwargs['xtick_rotation'],
				labelsize=kwargs['xtick_labelsize'],
			)
			ax.grid(ls=':')
		axes_inset.append(_axes_inset)
		axes.append(_axes)

		if kwargs['sup_text']:
			msg = kwargs['sup_text']
		else:
			msg = f"ci = {100 * ci} % whiskers: "
			msg += f"({100 * whis:0.2f}, {100 * (1 - whis):0.2f})\n"
			msg += "mean and scatter points: 'actual' sample\n"
			msg += "whiskers, caps: 'bootstrapped' samples"
		if add_suptitle:
			sup = fig.suptitle(msg, fontsize=kwargs['sup_fontsize'], y=kwargs['sup_y'])
		else:
			sup = None
		figs.append(fig)
		sups.append(sup)
		plt.close()
	if save_file:
		print(save_file)
		save_fig(
			fig=figs,
			sup=sups,
			save_file=save_file,
			display=False,
			multi=True,
		)
		print('done.')
	return figs, sups, axes, axes_inset
