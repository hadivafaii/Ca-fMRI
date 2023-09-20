from .fighelper import *
from model.mouse import BOLD, CAS, CAF


def scores_pie(
		results: dict,
		mu: float = 0.1,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'figsize': (10, 3),
		'palette': prepare_palette(),
		'grey': list(sns.color_palette('deep'))[-3],
		'textprops': {'color': 'w', 'fontsize': 18},
		'autopct': '%1.0f%%',
		'startangle': 0,
		'radius': {
			BOLD: (0.6, 0.48),
			CAS: (0.55, 0.5),
			CAF: (0.5, ) * 2},
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	figs = {}
	for i, (mode, d1) in enumerate(results['scores'].items()):
		fig, ax = create_figure(
			tight_layout=False,
			constrained_layout=False,
			figsize=kwargs['figsize'],
			dpi=kwargs['dpi'],
		)
		ovp = d1[mu]['score_nodes']
		*_, autotexts = ax.pie(
			x=[1 - ovp, ovp],
			autopct=kwargs['autopct'],
			textprops=kwargs['textprops'],
			startangle=kwargs['startangle'],
			colors=[kwargs['grey'], kwargs['palette'][mode]],
		)
		if kwargs['radius'] is not None:
			r = kwargs['radius'].get(mode)
			if not isinstance(r, Iterable):
				r = (r,) * len(autotexts)
			for r, t in zip(r, autotexts):
				xy = t.get_position()
				r0 = np.linalg.norm(xy)
				xy = [e * (r / r0) for e in xy]
				t.set_position(xy)
		figs[mode] = fig
		plt.close()
	return figs


def scores_bar(
		results: dict,
		mu: float = 0.1,
		legend: bool = False,
		display: bool = True,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'figsize': (10, 3),
		'palette': prepare_palette(),
		'annot_ratio_x': 10,
		'annot_delta_y': 2,
		'annot_fontsize': 9,
		'annot_rotation': 40,
		'tick_direction': 'out',
		'tick_labelsize': 12,
		'ylim': (0, 125),
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v
	fig, ax = create_figure(
		tight_layout=False,
		constrained_layout=False,
		figsize=kwargs['figsize'],
		dpi=kwargs['dpi'],
	)
	df = _bar_df(results)
	df = df.loc[df['mu'] == mu]
	sns.barplot(
		data=df,
		x='oc',
		y='score',
		hue='mode',
		palette=prepare_palette(),
		ax=ax
	)
	for rect in ax.patches:
		h = rect.get_height()
		w = rect.get_width()
		x = rect.get_x()
		xy = (
			x + w / kwargs['annot_ratio_x'],
			h + kwargs['annot_delta_y'],
		)
		ax.annotate(
			text=f"{h:0.0f}%",
			xy=xy,
			xycoords='data',
			fontsize=kwargs['annot_fontsize'],
			rotation=kwargs['annot_rotation'],
		)
	ax.set(
		ylim=kwargs['ylim'],
		yticks=[0, 100],
		ylabel='',
		xlabel='',
	)
	ax = set_tickparams(ax, **kwargs)
	if not legend:
		ax.get_legend().remove()
	ax.grid()

	if display:
		plt.show()
	else:
		plt.close()
	return fig, ax


def mk_member_cmap(
		spect: List[float] = None,
		acc: List[int] = None, ):
	spect = spect if spect else [0.20, 0.5, 0.75, 0.98]
	spectral = sns.color_palette('Spectral', as_cmap=True)
	spectral = matplotlib.colors.ListedColormap(
		colors=[spectral(i) for i in spect],
		name='members',
	)
	acc = acc if acc else [1, 2, 0, 5]
	accent = sns.color_palette('Accent')
	accent = matplotlib.colors.ListedColormap(
		colors=[accent[i] for i in acc],
		name='members',
	)
	return spectral, accent


def _bar_df(results: dict):
	dlist = []
	for mode, d1 in results['scores'].items():
		for mu, d2 in d1.items():
			scores = {
				f"OC{i + 1}": s * 100 for i, s
				in enumerate(d2['score_ocs'])
			}
			dlist.append({
				'k': [len(scores)] * len(scores),
				'mode': [mode] * len(scores),
				'mu': [mu] * len(scores),
				'oc': scores.keys(),
				'score': scores.values(),
			})
	df = pd.DataFrame(merge_dicts(dlist))
	return df
