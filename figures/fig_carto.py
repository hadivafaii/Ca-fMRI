from .fighelper import *


def cartography(
		df: pd.DataFrame,
		bt_dict: Dict[str, object],
		legend: bool = True,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (10, 4.2),
		'tval': get_tval(9, 0.95),
		'n_resamples': int(1e5),
		'line_lw': 5,
		'line_alpha': 0.8,
		'line_color': 'C7',
		'scatter_s': 25,
		'scatter_alpha': 0.9,
		'scatter_edgecolor': 'silver',
		'annot_xy': (0.05, 0.92),
		'annot_fontsize': 15,
		'legend_fontsize': 13,
		'legend_bbox': (1.025, 0.785),
		'tick_direction': 'in',
		'tick_labelsize': 12,
		'tick_nbins': 6,
		'ylim': (33, 160),
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	fig, axes = create_figure(
		nrows=1,
		ncols=3,
		tight_layout=False,
		constrained_layout=True,
		figsize=kwargs['figsize'],
	)
	axes[2].get_shared_y_axes().join(axes[2], axes[1])
	axes[2].get_shared_x_axes().join(axes[2], axes[1])
	for i, mode in enumerate(df['mode'].unique()):
		_df = df.loc[df['mode'] == mode]
		if i == 2:
			cond = (
					(_df['deg'] < 30) &
					(_df['ent'] > 0.6)
			)
			_df.loc[cond, 'ent'] = np.nan
			_df.loc[cond, 'deg'] = np.nan
		good = np.logical_and(
			np.isfinite(_df['ent']),
			np.isfinite(_df['deg']),
		)
		lr = sp_stats.linregress(
			x=_df['ent'][good],
			y=_df['deg'][good],
		)
		x_min = _df['ent'][good].min()
		x_max = _df['ent'][good].max()
		xs = np.linspace(x_min, x_max, 1000)
		ys = lr.slope * xs + lr.intercept

		ax = axes[i]
		ax.plot(
			xs, ys,
			lw=kwargs['line_lw'],
			alpha=kwargs['line_alpha'],
			color=kwargs['line_color'],
			zorder=0,
		)
		sns.scatterplot(
			data=_df,
			x='ent',
			y='deg',
			hue='oc_idx',
			legend=True if
			i == 2 else False,
			s=kwargs['scatter_s'],
			alpha=kwargs['scatter_alpha'],
			edgecolor=kwargs['scatter_edgecolor'],
			palette=bt_dict[mode].cmaps[-1],
			ax=ax,
		)
		ax = set_tickparams(ax, **kwargs)
		ax.set(xlabel='', ylabel='')
		ax.locator_params(
			axis='both',
			nbins=kwargs['tick_nbins'],
		)
		if i == 0:
			ax.set_ylim(kwargs['ylim'])
		elif i == 2:
			ax.set_yticklabels([])

		leg = ax.get_legend()
		if leg:
			if not legend:
				leg.remove()
			else:
				leg.set_title('OC')
				leg.set_bbox_to_anchor(kwargs['legend_bbox'])
				_ = plt.setp(
					obj=leg.get_texts(),
					fontsize=kwargs['legend_fontsize'],
				)
				_ = plt.setp(
					obj=leg.get_title(),
					fontsize=kwargs['legend_fontsize'] + 2,
				)
		rho, se = corr_group(
			x=bt_dict[mode].run_h,
			y=bt_dict[mode].run_deg,
			n_resamples=kwargs['n_resamples'],
		)
		text = r"$r = $" + f"{rho[0]:0.2f} " + r"$\pm$"
		text += f" {se * get_tval():0.2f}"
		ax.annotate(
			text=text,
			xy=kwargs['annot_xy'],
			xycoords='axes fraction',
			fontsize=kwargs['annot_fontsize'],
		)
	ax_square(axes)

	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes
