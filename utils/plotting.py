from .generic import *
import matplotlib
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import PathPatch
from matplotlib.path import Path as mpl_Path
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.colors import to_rgb, rgb2hex, Colormap, LinearSegmentedColormap


def move_legend(
		ax: plt.Axes,
		bbox: Tuple[float, float] = None, ):
	leg = ax.get_legend()
	if leg is None:
		return ax
	if bbox is None:
		leg.remove()
	else:
		leg.set_bbox_to_anchor(bbox)
	return ax


# noinspection PyUnresolvedReferences
def cbar_only(
		cmap,
		vmin: float = 0,
		vmax: float = 1,
		vertical: bool = True,
		**kwargs, ):
	defaults = {
		'figsize': (0.4, 4) if
		vertical else (4, 0.35),
		'edgecolor': 'k',
		'linewidth': 1.3,
		'tick_pad': 2,
		'tick_length': 6,
		'tick_labelsize': 12,
		'tick_position': 'right'
		if vertical else 'bottom',
	}
	for k, v in defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	fig, cax = create_figure(
		nrows=1,
		ncols=1,
		figsize=kwargs['figsize'],
		constrained_layout=False,
		tight_layout=False,
	)
	cbar = matplotlib.colorbar.ColorbarBase(
		ax=cax,
		cmap=matplotlib.cm.get_cmap(cmap)
		if isinstance(cmap, str) else cmap,
		norm=matplotlib.colors.Normalize(vmin, vmax),
		orientation='vertical' if vertical else 'horizontal',
	)

	cbar.outline.set_edgecolor(kwargs['edgecolor'])
	cbar.outline.set_linewidth(kwargs['linewidth'])

	cax.tick_params(
		axis='y' if vertical else 'x',
		pad=kwargs['tick_pad'],
		length=kwargs['tick_length'],
		labelsize=kwargs['tick_labelsize'],
		color=kwargs['edgecolor'],
		width=kwargs['linewidth'],
	)
	if vertical:
		cax.yaxis.set_ticks_position(kwargs['tick_position'])
	else:
		cax.xaxis.set_ticks_position(kwargs['tick_position'])
	plt.close()
	return fig, cax, cbar


def modify_box(ax, qs: List[Tuple[float, float, float, float]], axis: str = 'y'):
	children = ax.get_children()
	# fix box
	selected = [
		e for e in children if
		isinstance(e, PathPatch)
	]
	for q, pathpatch in zip(qs, selected):
		path = pathpatch.get_path()
		vertices = path.vertices.copy()
		vertices[0, 0 if axis == 'x' else 1] = q[1]
		vertices[1, 0 if axis == 'x' else 1] = q[1]
		vertices[2, 0 if axis == 'x' else 1] = q[2]
		vertices[3, 0 if axis == 'x' else 1] = q[2]
		vertices[4, 0 if axis == 'x' else 1] = q[2]
		pathpatch.set_path(mpl_Path(vertices, path.codes))
	# fix arms
	selected = [
		e for e in children
		if isinstance(e, Line2D)
	]
	selected = np.reshape(selected, (len(qs), -1))
	for i, q in enumerate(qs):
		selected[i, 0].set(**{'xdata' if axis == 'x' else 'ydata': (q[1], q[0])})
		selected[i, 1].set(**{'xdata' if axis == 'x' else 'ydata': (q[2], q[3])})
		selected[i, 2].set(**{'xdata' if axis == 'x' else 'ydata': (q[0], q[0])})
		selected[i, 3].set(**{'xdata' if axis == 'x' else 'ydata': (q[3], q[3])})
	return ax


def plot_last_bin(
		save_dir: str,
		booted: np.ndarray,
		df_real: pd.DataFrame,
		df_boot: pd.DataFrame,
		colors: Dict[int, str],
		dist_names: List[str] = None,
		figsize=(13, 9.5),
		ci: float = 99.9, ):
	from fitter import Fitter
	props_boot = {
		'boxprops': {'color': 'none', 'edgecolor': 'k', 'linewidth': 1},
		'medianprops': {'color': 'none', 'linewidth': 0},
		'whiskerprops': {'color': 'k', 'linewidth': 1},
		'capprops': {'color': 'k', 'linewidth': 1},
		'flierprops': {'alpha': 0.5, 'zorder': 1},
	}
	props_real = {
		'boxprops': {'color': 'none', 'edgecolor': 'none', 'linewidth': 0},
		'medianprops': {'color': 'k', 'linewidth': 1.5},
		'whiskerprops': {'color': 'none', 'linewidth': 0},
		'capprops': {'color': 'none', 'linewidth': 0},
		'flierprops': {'alpha': 0.7, 'zorder': 1},
	}
	meanprops = {
		"marker": "o",
		"markerfacecolor": "white",
		"markeredgecolor": "k",
		"markersize": "10",
	}
	dist_names = dist_names if dist_names else ['beta', 't']
	low = np.round((100 - ci) / 2, 10)
	high = np.round(100 - low, 10)

	intervals = df_real.interval.unique()
	k = len(df_real.oc_idx.unique())
	nrows = 2
	ncols = int(np.ceil(k / nrows))
	fig, axes = create_figure(
		nrows=nrows,
		ncols=ncols,
		figsize=figsize,
		style='ticks',
	)
	for i, ax in enumerate(axes.ravel()):
		f = Fitter(
			data=booted[:, i, -1],
			distributions=dist_names,
			bins=500,
			timeout=1000,
		)
		f.fit(progress=False)
		for _d in dist_names:
			pdf = make_pdf(
				dist=_d,
				params=f.fitted_param[_d],
				whis=(low / 100, high / 100),
				size=10000,
			)
			ax.plot(
				list(pdf.index),
				list(pdf.values),
				color='r' if _d == 't' else 'k',
				lw=1.5,
				label=_d,
			)
		ax.legend(loc='upper right')
		sns.histplot(
			booted[:, i, -1],
			color=colors[i + 1],
			stat='density',
			alpha=0.6,
			ax=ax,
		)
		ax.grid(ls=':')
		ax.set_title(f"OC{i + 1}")

		ax_tx = ax.twinx()
		_df = df_boot.loc[
			(df_boot.oc_idx == i + 1) &
			(df_boot.interval == intervals[-1])
			]
		sns.boxplot(
			data=_df,
			x='mean',
			whis=(low, high),
			showfliers=True,
			orient='h',
			width=0.15,
			ax=ax_tx,
			**props_boot,
		)

		_df = df_real.loc[
			(df_real.oc_idx == i + 1) &
			(df_real.interval == intervals[-1])
			]
		sns.boxplot(
			data=_df,
			x='mean',
			showmeans=True,
			meanprops=meanprops,
			showfliers=True,
			orient='h',
			width=0.15,
			ax=ax_tx,
			**props_real,
		)
		ax_tx.set_ylim((0.15, -1.0))
		if i // ncols == nrows - 1:
			ax.set_xlabel('mean portion')
		else:
			ax.set_xlabel('')
	plt.draw()
	for ax in axes.ravel():
		ax.set_xticklabels([
			np.round(float(e.get_text()), 2)
			for e in ax.get_xticklabels()
		])

	msg = f"ci = {ci} %  whiskers: [{low}, {high}]\n"
	msg += "median and mean (white circle): 'actual' sample\n"
	msg += "whiskers, caps, hist: 'bootstrapped' samples"
	sup = fig.suptitle(msg, fontsize=17, y=1.05)

	save_file = [
		'last-bin',
		f"bins-{len(intervals)}",
		f"boot-{len(booted):1.0e}",
	]
	save_file = f"{'_'.join(save_file)}.pdf"
	save_file = pjoin(save_dir, save_file)
	print(save_file)
	save_fig(
		fig=fig,
		sup=sup,
		save_file=save_file,
		display=False,
		multi=False,
	)
	print('done.')
	return fig, axes


def plot_bootstrap(
		save_dir: str,
		booted: np.ndarray,
		colors: Dict[int, str],
		intervals: List['str'],
		kind: str = 'kde',
		ci: List[int] = None, ):
	props = {
		'boxprops': {'alpha': 0.5, 'edgecolor': 'k'},
		'medianprops': {'color': 'k'},
		'whiskerprops': {'color': 'k'},
		'capprops': {'color': 'k', 'linewidth': 1},
		'flierprops': {'alpha': 0.5, 'zorder': 1},
	}
	meanprops = {
		"marker": "o",
		"markerfacecolor": "bisque",
		"markeredgecolor": "k",
		"markersize": "6",
	}
	ci = ci if ci else [95., 99., 99.9, 99.999]
	whisk = {
		c: (
			np.round((100 - c) / 2, 10),
			100 - np.round((100 - c) / 2, 10)
		) for c in ci
	}
	nrows = 2
	ncols = len(intervals)
	figs = []
	sups = []
	for ci, whisk_val in tqdm(whisk.items()):
		fig, axes = create_figure(
			nrows=nrows,
			ncols=ncols,
			figsize=(4 * ncols, 3.8 * nrows),
			style='ticks',
			sharex='col',
			sharey='none',
		)
		for j in range(ncols):
			dlist = []
			for i in range(booted.shape[-2]):
				dlist.append({
					'oc_idx': [i + 1] * len(booted),
					'mean': booted[:, i, j],
				})
			_df = pd.DataFrame(merge_dicts(dlist))
			if kind == 'kde':
				sns.kdeplot(
					data=_df,
					x='mean',
					hue='oc_idx',
					fill=True,
					alpha=0.05,
					lw=2,
					palette=colors,
					ax=axes[0, j],
				)
			elif kind == 'hist':
				sns.histplot(
					data=_df,
					x='mean',
					hue='oc_idx',
					alpha=0.5,
					bins=100,
					palette=colors,
					ax=axes[0, j],
				)
			else:
				raise RuntimeError(f"invalid kind: '{kind}'")
			axes[0, j].set_title(f"bin: {intervals[j]}\n", fontsize=15)

			sns.boxplot(
				data=_df,
				x='mean',
				y='oc_idx',
				palette=colors,
				dodge=True,
				orient='h',
				width=0.6,
				whis=whisk_val,
				fliersize=1,
				showfliers=False,
				showmeans=True,
				meanprops=meanprops,
				ax=axes[1, j],
				**props,
			)
			for i in range(2):
				axes[i, j].grid(axis='both', ls=':')
				axes[i, j].axvline(0, color='dimgrey', lw=0.8, ls='--')
			if j == 0:
				axes[0, j].set_ylabel('kde (mice)\n', fontsize=15)
				axes[1, j].set_ylabel('box (mice)\n', fontsize=15)
			else:
				for i in range(2):
					axes[i, j].set_ylabel('')

		msg = f"Whiskers at {ci} % confidence interval: {whisk_val}"
		sup = fig.suptitle(msg, fontsize=20, y=1.07)
		figs.append(fig)
		sups.append(sup)
		plt.close()

	save_file = [
		kind,
		f"bins-{len(intervals)}",
		f"boot-{len(booted):1.0e}",
	]
	save_file = f"{'_'.join(save_file)}.pdf"
	save_file = pjoin(save_dir, save_file)
	print(save_file)
	save_fig(
		fig=figs,
		sup=sups,
		save_file=save_file,
		display=False,
		multi=True,
	)
	print('done.')
	return figs, sups


def add_ax_inset(
		ax: plt.Axes,
		data: np.ndarray,
		bounds: List[float],
		kind: str = 'imshow',
		aspect_eq: bool = True,
		**kwargs, ):
	axins = ax.inset_axes(bounds)
	if kind == 'imshow':
		x2p = np.ma.masked_where(data[0] == 0, data[0])
		axins.imshow(x2p, cmap='Greys_r')
		x2p = np.ma.masked_where(data[1] < kwargs['vmin'], data[1])
		axins.imshow(x2p, **kwargs)
	elif kind == 'kde':
		sns.kdeplot(data=data, ax=axins, **kwargs)
	else:
		raise NotImplementedError
	if aspect_eq:
		axins.set_aspect('equal', adjustable='box')
	return axins


def plot_annot_2d(
		x: np.ndarray,
		cticks: List[float],
		ctick_labels: List[str],
		fig: plt.figure = None,
		ax: plt.axes = None,
		**kwargs, ):

	kwarg_defaults = {
		'ctick_labelsize': 12,
		'ctick_length': 7,
		'ctick_width': 1.3,
		'shrink': 0.9,
		'pad': 0.04,
		'figsize': (10, 7),
		'cmap': 'tab10',
		'tight_layout': True,
	}
	for k, v in kwarg_defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	if fig is None or ax is None:
		fig, ax = create_figure(
			nrows=1,
			ncols=1,
			figsize=kwargs['figsize'],
			tight_layout=kwargs['tight_layout'],
		)

	vmin = np.min(x) - 0.5
	vmax = np.max(x) + 0.5
	im = ax.imshow(x, cmap=kwargs['cmap'], vmin=vmin, vmax=vmax)

	cbar = fig.colorbar(
		mappable=im,
		ticks=cticks,
		pad=kwargs['pad'],
		shrink=kwargs['shrink'],
	)
	cbar.ax.set_yticklabels(ctick_labels)
	cbar.ax.tick_params(
		labelsize=kwargs['ctick_labelsize'],
		length=kwargs['ctick_length'],
		width=kwargs['ctick_width'],
	)
	remove_ticks(ax)
	if kwargs['tight_layout']:
		fig.tight_layout()
	return fig, ax


def plot_registration(fixed, moving, warped, sup: bool = False, figsize=(4, 5)):
	fig, axes = create_figure(3, 3, figsize)

	for i in range(3):
		for j in range(3):
			if i == 0:
				x = fixed.numpy()
				label = 'fixed'
			elif i == 1:
				x = moving.numpy()
				label = 'moving'
			else:
				x = warped.numpy()
				label = 'warped'

			if i == 0:
				if j == 0:
					top_title = 'axis = {:d}\n(coronal)'.format(j)
				elif j == 1:
					top_title = 'axis = {:d}\n(axial)'.format(j)
				else:
					top_title = 'axis = {:d}\n(saggital)'.format(j)
				axes[i, j].set_title(top_title, y=1.1, fontsize=13)

			if j == 0:
				axes[i, j].set_ylabel(label + '\n', fontsize=13)

			data2plt = x.mean(j)
			axes[i, j].imshow(data2plt)

			_ = list(map(lambda z: z.set_visible(False), axes[i, j].spines.values()))
			axes[i, j].set_xticks([])
			axes[i, j].set_yticks([])

	if sup:
		msg = 'Plotting averaged data across different dimensions\n\n'
		msg += f"1st row (fixed):  Average template,    shape = {fixed.shape}\n"
		msg += f"2nd row (moving):  Anatomical data,    shape = {moving.shape}\n"
		msg += f"3rd row (warped):  Anatomical data,    shape = {warped.shape}"
		plt.suptitle(msg, fontsize=14, y=1.02)
	plt.tight_layout()
	plt.show()

	return fig, axes


def plot_nx_tree(tree: dict, prog: str = 'neato', **kwargs):
	kwargs_defaults = {
		'arrowsize': 20 if prog == 'neato' else 5,
		'node_size': 4200 if prog == 'neato' else 1200,
		'font_size': 11 if prog == 'neato' else 10,
		'figsize': (13.5, 10) if prog == 'neato' else (13.5 * 6, 10),
		'arrowstyle': '->',
		'width': 1.5,
	}
	for k, v in kwargs_defaults.items():
		if k not in kwargs:
			kwargs[k] = v

	fig, ax = create_figure(1, 1, figsize=kwargs['figsize'])
	pos = graphviz_layout(tree['graph'], prog=prog, root=list(tree['graph'].nodes)[0])
	nx.draw(
		G=tree['graph'],
		pos=pos,
		with_labels=True,
		arrows=True,
		arrowsize=kwargs['arrowsize'],
		arrowstyle=kwargs['arrowstyle'],
		width=kwargs['width'],
		node_size=kwargs['node_size'],
		font_size=kwargs['font_size'],
		edge_color='k',
		node_color=tree['node2c'].values(),
		ax=ax,
	)
	fig.tight_layout()
	plt.show()

	return fig, ax


def plot_corr_hist(
		corr_list: List[np.ndarray],
		labels: List[str],
		colors: List[str] = None,
		positive_only: bool = True,
		cmap: str = 'tab10',
		figsize=(12, 8), ):

	if colors is None:
		colors = list(sns.color_palette(cmap, n_colors=len(corr_list)))

	sns.set_style('whitegrid')
	fig, ax = plt.subplots(1, 1, figsize=figsize)
	for item, lbl, color in zip(corr_list, labels, colors):
		x = np.triu(item)
		x = x.flatten()
		data2plt = x[x > 0.0] if positive_only else x[x != 0]
		sns.histplot(
			data2plt, kde=True, color=color, alpha=0.25,
			stat='probability', line_kws={'lw': 3}, label=lbl, ax=ax,
		)
		mean = data2plt.mean()
		label = '{:s} mean: {:.3f}'
		label = label.format(lbl, mean)
		ax.axvline(mean, color=color, ls='--', lw=1.5, label=label)

	ax.set_title('Histogram plot of correlations', fontsize=15, y=1.02)
	ax.legend(loc='upper left')
	plt.show()
	return fig, ax


def plot_corrs(
		pearson: Dict[int, np.ndarray],
		spearman: Dict[int, np.ndarray],
		figsize=None, dpi=70, ):
	percentiles = set(pearson.keys()).intersection(spearman.keys())
	percentiles = sorted(list(percentiles), reverse=True)

	if figsize is None:
		figsize = (13.5, 5.5 - np.ceil(len(percentiles) / 6))

	sns.set_style('white')
	fig, axes = plt.subplots(2, len(percentiles), figsize=figsize, dpi=dpi)

	for i, perc in enumerate(percentiles):
		data2plot = spearman[perc]
		vmin = np.min(data2plot[data2plot > 0])
		sns.heatmap(
			data2plot,
			cmap='nipy_spectral',
			vmin=vmin,
			vmax=1.,
			cbar_kws={'aspect': 5 * len(percentiles), 'shrink': 0.8},
			ax=axes[0, i],
		)
		axes[0, i].set_aspect('equal')
		axes[0, i].set_xticks([])
		axes[0, i].set_yticks([])
		msg = 'top {:.1f}{:s}\n'
		msg = msg.format((pearson[perc] > 0).sum() / len(pearson[perc]) ** 2 * 100, '%')
		axes[0, i].set_title(msg, fontsize=15)

		data2plot = spearman[perc]
		vmin = np.min(data2plot[data2plot > 0])
		sns.heatmap(
			data2plot,
			cmap='nipy_spectral',
			vmin=vmin,
			vmax=1.,
			cbar_kws={'aspect': 5 * len(percentiles), 'shrink': 0.8},
			ax=axes[1, i],
		)
		axes[1, i].set_aspect('equal')
		axes[1, i].set_xticks([])
		axes[1, i].set_yticks([])

		if i == 0:
			axes[0, i].set_ylabel('pearson', fontsize=15)
			axes[1, i].set_ylabel('spearman', fontsize=15)

	msg = 'Correlation mats using different thresholds.  1st row: pearson, 2nd row: spearman'
	fig.suptitle(msg, y=1.03, fontsize=17)
	fig.tight_layout()
	plt.show()

	return fig, axes


def quick_viz(g, a, num_comm: int = None, figsize=(10, 8)):
	sns.set_style('whitegrid')
	fig, axes = plt.subplots(2, 2, figsize=figsize)

	# weight hist
	data = a.copy()
	data = data[data > 0]
	sns.histplot(data, kde=True, ax=axes[0, 0])
	axes[0, 0].set_title('(nonzero) weight distribution', fontsize=15, y=1.03)
	axes[0, 0].set_xlabel('W', fontsize=11)
	axes[0, 0].set_ylabel('Count', fontsize=11)

	# eig and ipr
	w, v = np.linalg.eigh(nx.normalized_laplacian_matrix(g).toarray())
	ipr = 1 / np.power(v, 4).sum(0)
	ipr /= len(ipr)
	sort_indxs = np.argsort(w)

	num_comm = 10 if num_comm is None else num_comm
	axes[0, 1].scatter(
		w[sort_indxs[num_comm:]], ipr[sort_indxs[num_comm:]],
		marker='x', color='k', s=130, alpha=0.4,
	)
	axes[0, 1].scatter(
		w[sort_indxs[:num_comm]], ipr[sort_indxs[:num_comm]],
		marker='o', edgecolors='k', lw=2, color='none', s=130,
	)
	msg = 'Scatter plot of eigenvalues vs. IPR  (# comm = {:d})'
	axes[0, 1].set_title(msg.format(num_comm), fontsize=15, y=1.03)
	axes[0, 1].set_xlabel('Eigenvalues', fontsize=11)
	axes[0, 1].set_ylabel('IPR', fontsize=11)

	# degree distribution
	nodes, degrees = list(zip(*g.degree()))
	nodes, weighted_degrees = list(zip(*g.degree(weight='weight')))
	data_dict = {
		'node': nodes,
		'deg': degrees,
		'deg_w': weighted_degrees,
	}
	deg_df = pd.DataFrame.from_dict(data_dict)

	sns.kdeplot(data=deg_df, x='deg_w', color='coral', lw=4, alpha=0.8, ax=axes[1, 0])
	sns.kdeplot(data=deg_df, x='deg', color='slategrey', lw=4, alpha=0.8, ax=axes[1, 1])

	_mean = deg_df.deg_w.mean()
	axes[1, 0].axvline(_mean, ls=':', lw=2, color='k', label='mean (weighted):  {:.1f}'.format(_mean))
	_mean = deg_df.deg.mean()
	axes[1, 1].axvline(_mean, ls=':', lw=2, color='k', label='mean (unweighted):  {:.1f}'.format(_mean))

	axes[1, 0].set_title('weighted deg dist', fontsize=15, y=1.02)
	axes[1, 1].set_title('unweighted deg dist', fontsize=15, y=1.02)

	for i in range(2):
		axes[1, i].set_xlabel('Degree', fontsize=11)
		axes[1, i].set_ylabel('Count', fontsize=11)
		axes[1, i].legend()

	msg = '# links = {:d},  # nodes = {:d},  m / n = {:.1f},  density = 2m / n(n-1) = {:.1f}'
	msg = msg.format(len(g.edges()), g.order(), len(g.edges()) / g.order(), nx.density(g))
	fig.suptitle(msg, fontsize=15, y=1.01)
	fig.tight_layout()
	plt.show()


def community_layout(g: nx.Graph(), partition: Dict[int, int]):
	"""
	Compute the layout for a modular graph.

	Arguments:
	----------
	g -- networkx.Graph or networkx.DiGraph instance
		graph to plot

	partition -- dict mapping int node -> int community
		graph partitions


	Returns:
	--------
	pos -- dict mapping int node -> (float x, float y)
		node positions

	"""

	pos_communities = _position_communities(g, partition, scale=3.)
	pos_nodes = _position_nodes(g, partition, scale=1.)

	# combine positions
	pos = dict()
	for node in g.nodes():
		pos[node] = pos_communities[node] + pos_nodes[node]

	return pos


def _position_communities(g, partition, **kwargs):
	# create a weighted graph, in which each node corresponds to a community,
	# and each edge weight to the number of edges between communities
	between_community_edges = _find_between_community_edges(g, partition)

	communities = set(partition.values())
	hypergraph = nx.DiGraph()
	hypergraph.add_nodes_from(communities)
	for (ci, cj), edges in between_community_edges.items():
		hypergraph.add_edge(ci, cj, weight=len(edges))

	# find layout for communities
	pos_communities = nx.spring_layout(hypergraph, **kwargs)

	# set node positions to position of community
	pos = dict()
	for node, community in partition.items():
		pos[node] = pos_communities[community]

	return pos


def _find_between_community_edges(g, partition):
	edges = dict()

	for (ni, nj) in g.edges():
		ci = partition[ni]
		cj = partition[nj]

		if ci != cj:
			try:
				edges[(ci, cj)] += [(ni, nj)]
			except KeyError:
				edges[(ci, cj)] = [(ni, nj)]

	return edges


def _position_nodes(g, partition, **kwargs):
	"""
	Positions nodes within communities.
	"""

	communities = dict()
	for node, community in partition.items():
		try:
			communities[community] += [node]
		except KeyError:
			communities[community] = [node]

	pos = dict()
	for ci, nodes in communities.items():
		subgraph = g.subgraph(nodes)
		pos_subgraph = nx.spring_layout(subgraph, **kwargs)
		pos.update(pos_subgraph)

	return pos


def plot_graph(
		g: nx.Graph(),
		partition_pred: Dict[int, int],
		partition_true: Dict[int, int] = None,
		cmap: str = None,
		figsize=(13.5, 6), ):

	if partition_true is not None:
		pos = community_layout(g, partition_true)
	else:
		pos = community_layout(g, partition_pred)

	num_comm_true = len(set(partition_true.values()))
	num_comm_pred = len(set(partition_pred.values()))
	num_comm_max = max(num_comm_true, num_comm_pred)

	if cmap is None:
		if num_comm_max <= 8:
			cmap = 'Accent'
		elif num_comm_max <= 12:
			cmap = 'Set3'
		else:
			cmap = 'tab20'

	# TODO: make this fig, axes kinda thing, and when partition true is None sensitive
	plt.figure(figsize=figsize)

	plt.subplot(121)
	plt.axis('off')
	nx.draw_networkx_nodes(
		g, pos, node_size=130, edgecolors='k',
		cmap=cmap, node_color=list(partition_true.values()))
	nx.draw_networkx_edges(g, pos, alpha=0.1)
	msg = 'True ({:d} communities)'.format(num_comm_true)
	plt.title(msg, fontsize=15)

	plt.subplot(122)
	plt.axis('off')
	nx.draw_networkx_nodes(
		g, pos, node_size=130, edgecolors='k',
		cmap=cmap, node_color=list(partition_pred.values()))
	nx.draw_networkx_edges(g, pos, alpha=0.1)
	msg = 'Pred ({:d} communities)'.format(num_comm_pred)
	plt.title(msg, fontsize=15)

	plt.tight_layout()
	plt.show()

	return


def plot_louvain_results(
		results: Dict[str, dict],
		selected_idxs: List[int] = None,
		partition_true: Dict[int, int] = None, ):

	selected_idxs = [] if selected_idxs is None else selected_idxs
	palette = sns.color_palette('autumn_r', n_colors=len(selected_idxs))

	sns.set_style('whitegrid')
	if partition_true is None:
		fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), sharex='all')
		axes = np.reshape(axes, (1, -1))
	else:
		fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex='all')

	x0, y0 = tuple(zip(*results['modularities'].items()))
	axes[0, 0].semilogx(x0, y0, lw=3)
	max_mod_idx = np.argmax(y0)
	max_mod = y0[max_mod_idx]
	res_at_max_mod = x0[max_mod_idx]
	lbl = 'best idx = {:d},   max Q = {:.3f},   at res = {:.2f}'
	lbl = lbl.format(max_mod_idx, max_mod, res_at_max_mod)
	axes[0, 0].axvline(res_at_max_mod, color='k', lw=1.5, ls='--', label=lbl)

	for i, _idx in enumerate(selected_idxs):
		lbl = 'idx = {:d},   selected Q = {:.3f},   at res = {:.2f}'.format(_idx, y0[_idx], x0[_idx])
		axes[0, 0].axvline(x0[_idx], color=list(palette)[i], lw=2, ls=':', label=lbl)

	x1, y1 = tuple(zip(*results['num_communities'].items()))
	axes[0, 1].loglog(x1, y1, lw=3)
	lbl = '# comm = {:d}'.format(y1[max_mod_idx])
	axes[0, 1].axvline(res_at_max_mod, color='k', lw=2, ls='--', label=lbl)

	for i, _idx in enumerate(selected_idxs):
		lbl = '# comm = {:d}'.format(y1[_idx])
		axes[0, 1].axvline(x0[_idx], color=list(palette)[i], lw=2, ls=':', label=lbl)

	axes[0, 0].set_title('Q as a function of resolution', fontsize=15, y=1.02)
	axes[0, 0].set_xlabel('resolution (log)', fontsize=12)
	axes[0, 0].set_ylabel('Q (modularity)', fontsize=12)
	axes[0, 0].legend(loc='lower left')

	axes[0, 1].set_title('Num communities as a function of resolution', fontsize=15, y=1.02)
	axes[0, 1].set_xlabel('resolution (log)', fontsize=12)
	axes[0, 1].set_ylabel('# communities (log)', fontsize=12)
	axes[0, 1].legend(loc='upper right')

	if partition_true is not None:
		# scores = {res: compare_partitions(partition_true, p) for res, p in results['partitions'].items()}
		scores = {}
		x2, y2 = tuple(zip(*scores.items()))
		ami, nmi, _ = zip(*y2)

		axes[1, 0].semilogx(x2, nmi, lw=1.5, color='magenta', label='NMI')
		axes[1, 0].semilogx(x2, ami, lw=1.5, color='g', label='AMI')
		lbl = 'NMI = {:.1f}'.format(nmi[max_mod_idx])
		axes[1, 0].axvline(res_at_max_mod, color='k', lw=2, ls='--', label=lbl)

		for i, _idx in enumerate(selected_idxs):
			lbl = 'NMI = {:.1f}'.format(nmi[_idx])
			axes[1, 0].axvline(x0[_idx], color=list(palette)[i], lw=2, ls=':', label=lbl)

		axes[1, 0].set_title('Normalized/adjusted MI as a function of resolution', fontsize=15, y=1.02)
		axes[1, 0].set_xlabel('resolution (log)', fontsize=12)
		axes[1, 0].set_ylabel('Score', fontsize=12)
		axes[1, 0].legend(loc='upper right')

	fig.tight_layout()
	plt.show()

	return fig, axes


def add_jitter(
		x: np.ndarray,
		sigma: float = 0.01,
		shift_mean: bool = True, ):
	jit = get_rng().normal(scale=sigma, size=len(x))
	if shift_mean:
		jit -= jit.mean()
	return x + jit


def _iter_ax(axes):
	if not isinstance(axes, Iterable):
		return [axes]
	elif isinstance(axes, np.ndarray):
		return axes.flat


def ax_square(axes):
	for ax in _iter_ax(axes):
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		delta_x = max(xlim) - min(xlim)
		delta_y = max(ylim) - min(ylim)
		aspect = delta_x / delta_y
		ax.set_aspect(aspect, adjustable='box')
	return axes


def add_grid(axes):
	for ax in _iter_ax(axes):
		ax.grid()
	return axes


def remove_ticks(axes, full: bool = True):
	for ax in _iter_ax(axes):
		if full:
			try:
				_ = list(map(
					lambda z: z.set_visible(False),
					ax.spines.values()
				))
			except AttributeError:
				continue
		ax.set_xticks([])
		ax.set_yticks([])
	return


def set_style(
		context: str = 'notebook',
		style: str = 'ticks',
		palette: str = None,
		font: str = 'sans-serif', ):
	sns.set_theme(
		context=context,
		style=style,
		palette=palette,
		font=font,
	)
	matplotlib.rcParams['grid.linestyle'] = ':'
	matplotlib.rcParams['figure.figsize'] = (5.0, 3.2)
	matplotlib.rcParams['image.interpolation'] = 'none'
	matplotlib.rcParams['font.family'] = font
	return


def create_figure(
		nrows: int = 1,
		ncols: int = 1,
		figsize: Tuple[float, float] = None,
		sharex: str = 'none',
		sharey: str = 'none',
		layout: str = None,
		wspace: float = None,
		hspace: float = None,
		width_ratios: List[float] = None,
		height_ratios: List[float] = None,
		reshape: bool = False,
		style: str = 'ticks',
		dpi: float = None,
		**kwargs, ):
	"""
	:param nrows:
	:param ncols:
	:param figsize:
	:param layout: {'constrained', 'compressed', 'tight', None}
	:param sharex: {'none', 'all', 'row', 'col'} or bool
	:param sharey: {'none', 'all', 'row', 'col'} or bool
	:param style: {'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'}
	:param wspace:
	:param hspace:
	:param width_ratios:
	:param height_ratios:
	:param reshape:
	:param dpi:
	:param kwargs:
	:return: fig, axes
	"""
	set_style(style=style)
	figsize = figsize if figsize else plt.rcParams.get('figure.figsize')
	dpi = dpi if dpi else plt.rcParams.get('figure.dpi')

	fig, axes = plt.subplots(
		nrows=nrows,
		ncols=ncols,
		sharex=sharex,
		sharey=sharey,
		layout=layout,
		figsize=figsize,
		gridspec_kw={
			'wspace': wspace,
			'hspace': hspace,
			'width_ratios': width_ratios,
			'height_ratios': height_ratios},
		dpi=dpi,
		**kwargs,
	)
	if nrows * ncols > 1 and reshape:
		axes = np.reshape(axes, (nrows, ncols))
	return fig, axes


def make_cmap(
		ramp_colors: List[str],
		name: str = 'custom_cmap',
		n_colors: int = 256,
		show: bool = True, ):
	color_ramp = LinearSegmentedColormap.from_list(
		name=name,
		colors=[to_rgb(c) for c in ramp_colors],
		N=n_colors,
	)
	if show:
		display_cmap(color_ramp, len(ramp_colors))
	return color_ramp


def get_affiliation_cmap(
		use_allen: bool = False,
		colors: List[str] = None,
		return_clist: bool = False, ):
	if use_allen:
		colors = [
			'#188064', '#019298', '#08848b',
			'#40a566', '#aec0b6', '#849b9a',
		]
	else:
		colors = colors if colors else [
			'#1aa2ef', '#4229d9', '#ea1525',
			'#f29621', '#f0f400', '#a540a0',
		]
	cmap = make_cmap(
		ramp_colors=colors,
		n_colors=len(colors),
		name='Allen',
		show=False,
	)
	if return_clist:
		return cmap, colors
	else:
		return cmap


def get_rand_cmap(num: int, rng=None):
	rng = rng if rng else get_rng()
	colors = (
		rng.choice(256, size=num) / 256,
		rng.choice(256, size=num) / 256,
		rng.choice(256, size=num) / 256,
	)
	colors = [
		rgb2hex(c) for c in
		list(zip(*colors))
	]
	cmap = make_cmap(
		ramp_colors=colors,
		name='random',
		n_colors=num,
		show=False,
	)
	return cmap


def get_hm_cmap(
		colors: List[str] = None,
		return_clist: bool = False, ):
	colors = colors if colors else [
		'#365c7f', '#09f6e1', '#7ac837',
		'#fbf504', '#f0640f', '#f50a10',
	]
	heatmap = make_cmap(
		ramp_colors=colors,
		name='heatmap',
		show=False,
	)
	if return_clist:
		return heatmap, colors
	else:
		return heatmap


def get_cluster_cmap(
		color_ids: Union[int, Sequence[int]] = None,
		return_colors: bool = False, ):
	ramp_colors = [
		'#3d6af5', '#f5af3d', '#3df554',
		'#f5443d', '#a03df5', '#930e7a',
		'#00d9ff', '#056133', '#f53d91',
		'#3df5f1', '#f5603d', '#ff96c7',
		'#c85506', '#edf53d', '#f53def',
		'#f5a37a', '#7a89f5', '#f5aeab',
		'#c4f5f4', '#3d46f5', '#050a61',
		'#355551', '#3df599', '#4b4d40',
		'#8f6f90', '#af506c', '#00d050',
		'#41adbe', '#8E93EB', '#7e4cb3',
		'#61052e',
	]
	if color_ids is None:
		cluster_cmap = make_cmap(
			ramp_colors=ramp_colors,
			n_colors=len(ramp_colors),
			name='cluster',
			show=False,
		)
		if return_colors:
			return cluster_cmap, np.array(ramp_colors)
		else:
			return cluster_cmap
	elif isinstance(color_ids, int):
		color = ramp_colors[color_ids]
		cluster_cmap = sns.light_palette(
			color=color,
			n_colors=2,
			as_cmap=True,
		)
		if return_colors:
			return cluster_cmap, color
		else:
			return cluster_cmap
	else:
		colors = np.array(ramp_colors)[color_ids]
		cluster_cmap = make_cmap(
			ramp_colors=colors,
			n_colors=len(colors),
			name='cluster',
			show=False,
		)
		if return_colors:
			return cluster_cmap, colors
		else:
			return cluster_cmap


def display_cmap(cmap: Union[str, Colormap], num_colors: int = 256):
	plt.figure(figsize=(13.5, 3))
	plt.imshow(
		[list(np.arange(0, num_colors, 0.11 * num_colors / 4))],
		interpolation='nearest',
		origin='lower',
		cmap=cmap,
	)
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
	plt.show()
	return


def cmap2pal(cmap: Colormap, n_colors: int, start: float):
	x = np.linspace(start % 1, 1.0, n_colors)
	x = [cmap(abs(i - 1e-10)) for i in x]
	return sns.color_palette(x)


def trim_axs(axes, n):
	axs = axes.flat
	for ax in axs[n:]:
		ax.remove()
	return axs[:n]


def complement_color(r, g, b):
	k = _hilo(r, g, b)
	return rgb2hex(tuple(k - u for u in (r, g, b)))


def _hilo(a, b, c):
	if c < b:
		b, c = c, b
	if b < a:
		a, b = b, a
	if c < b:
		b, c = c, b
	return a + c


def get_fonts_html():
	import matplotlib.font_manager

	def _mk_html(fontname):
		html = "<p>{font}: <span style='font-family:{font}; "
		html += "font-size: 24px;'>{font}  01234 </p>"
		return html.format(font=fontname)

	code = sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
	code = [_mk_html(font) for font in code]
	code = "\n".join(code)
	return "<div style='column-count: 2;'>{}</div>".format(code)


def save_fig(
		fig: Union[plt.Figure, List[plt.Figure]],
		sup: Union[plt.Text, List[plt.Text]],
		save_file: str,
		display: bool = False,
		multi: bool = False,):

	if save_file is not None:
		save_dir = os.path.dirname(save_file)
		try:
			os.makedirs(save_dir, exist_ok=True)
		except FileNotFoundError:
			pass
		if not multi:
			fig.savefig(save_file, dpi=fig.dpi, bbox_inches='tight', bbox_extra_artists=[sup])
		else:
			assert len(fig) == len(sup) > 1, "must provide > 1 figs for multi saving"
			with PdfPages(save_file) as pages:
				for f, s in zip(fig, sup):
					if f is None:
						continue
					canvas = FigureCanvasPdf(f)
					if s is not None:
						canvas.print_figure(pages, dpi=f.dpi, bbox_inches='tight', bbox_extra_artists=[s])
					else:
						canvas.print_figure(pages, dpi=f.dpi, bbox_inches='tight')
	if display:
		if isinstance(fig, list):
			for f in fig:
				plt.show(f)
		else:
			plt.show(fig)
	else:
		if isinstance(fig, list):
			for f in fig:
				plt.close(f)
		else:
			plt.close(fig)
