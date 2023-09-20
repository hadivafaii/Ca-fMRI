from utils.generic import *
import networkx as nx
from treelib import Tree
from prettytable import PrettyTable
from bg_atlasapi import BrainGlobeAtlas
from bg_space import AnatomicalSpace
from allensdk.core.mouse_connectivity_cache \
	import MouseConnectivityCache, MouseConnectivityApi
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from io import StringIO


class Allen(object):
	def __init__(
			self,
			resolution: int = 100,
			proj_mode: str = 'top_view',
			full_load: bool = False,
			ws_dir: str = 'Documents/workspaces/allen',
	):
		super(Allen, self).__init__()

		self.ws_dir = pjoin(
			os.environ['HOME'], ws_dir, 'MouseConnectivity')
		self.manifest_file = pjoin(
			self.ws_dir, 'manifest.json')
		self.mcc = MouseConnectivityCache(
			resolution=resolution,
			manifest_file=self.manifest_file,
			ccf_version=MouseConnectivityApi.CCF_2017,
		)
		self.s_tree = self.mcc.get_structure_tree()
		self.df = pd.DataFrame(self.s_tree.nodes())
		self.proj_mode = proj_mode
		self._setup_meta()
		self._setup_lr()

		self.bg = load_bg_atlas(resolution)
		self.nissl = None
		self.template = None
		self.annotation = None
		self.lookup = None
		self.paths = None
		if full_load:
			self.load_everything(resolution)

	def __call__(self, item):
		return self.get(item)

	def __getitem__(self, item):
		if self.template is None:
			return
		return self.template[item]

	def __repr__(self):
		return self.bg.__repr__()

	def __str__(self):
		buf = StringIO()
		_console = Console(
			file=buf,
			force_jupyter=False,
		)
		_console.print(self)
		return buf.getvalue()

	def __rich_console__(self, *args):
		panel = _rich_atlas_metadata(
			atlas_name=self.bg.atlas_name,
			metadata=self.bg.metadata,
		)
		yield panel

	def load_everything(self, resolution: int = 100, proj_mode: str = None):
		annotation, _ = self.mcc.get_annotation_volume()
		template, _ = self.mcc.get_template_volume()
		self.annotation = annotation.astype('uint32')
		self.template = template.astype(float)
		self.nissl = self.load_nissl(resolution)
		self.lookup, self.paths = self.load_paths(proj_mode)
		return

	def get(
			self,
			query: Union[int, str, List[Union[int, str]]],
	) -> Union[dict, pd.DataFrame]:
		if isinstance(query, str) or not isinstance(query, Iterable):
			query = [query]
		if all(isinstance(e, (int, np.int64, np.uint32)) for e in query):
			x = self.df.loc[self.df['id'].isin(query)]
		elif all(isinstance(e, str) for e in query):
			x = self.df.loc[self.df['acronym'].isin(query)]
		else:
			raise RuntimeError(f"invalid query: '{query}'")
		if len(x) == 1:
			return x.iloc[0].to_dict()
		return x

	def get_masks(
			self,
			query: Union[int, str, List[Union[int, str]]],
			as_objectmap: bool = False, ) -> Union[np.ndarray, Dict[str, np.ndarray]]:

		x = self.get(query)
		if isinstance(x, dict):
			mask = self.mcc.get_structure_mask(x['id'])[0]
			if as_objectmap:
				return mask.astype('uint32') * x['id']
			else:
				return mask.astype(bool)
		masks = {
			k: self.mcc.get_structure_mask(i)[0].astype('uint32') * i
			for k, i in dict(zip(x['acronym'], x['id'])).items()
		}
		if as_objectmap:
			return sum(e for e in masks.values())
		else:
			return masks

	def get_subtree(self, base_id: int, n_levels: int = None, show: bool = True):
		if not hasattr(self, 'bg'):
			self.bg = load_bg_atlas(self.mcc.resolution)
		base_id = base_id if base_id else self.bg.hierarchy.root
		base_identifier = self.bg.hierarchy.nodes[base_id].identifier
		base_lvl = self.bg.hierarchy.level(base_identifier)
		subtree = self.bg.hierarchy.subtree(base_id)
		n_levels = n_levels if n_levels else subtree.depth() + 1
		if show:
			subtree.show(filter=_get_tree_filter_fn(subtree, n_levels))
		return subtree, base_lvl

	def get_major_12(self, further: Dict[str, list] = None):
		further = further if further else {
			'HPF': ['HIP', 'RHP'],
		}
		_, df = self.load_info_dfs()
		major = 'Brain - Major Divisions'
		major = df.loc[df['name'] == major]['id'].item()
		major = self.s_tree.get_structures_by_set_id([major])
		major = [item['acronym'] for item in major]
		# further breakdown
		for src, dst_list in further.items():
			try:
				i = major.index(src)
			except ValueError:
				continue
			for item in dst_list:
				major.insert(i, item)
				i += 1
			major.pop(i)
		return major

	def get_major_structs(self):
		structs = [
			'fiber tracts', 'VS',
			'CB', 'HB', 'IB', 'MB',
			'HIP', 'RHP', 'OLF',
			'CTXsp', 'PAL', 'STR',
		]
		cortical = []
		subtree, _ = self.get_subtree(315, 2, show=False)
		for node in subtree.nodes.values():
			if subtree.level(node.identifier) == 1:
				cortical.append(node.identifier)
		structs += self.get(cortical)['acronym'].values.tolist()
		return structs

	def fuse_masks(
			self,
			struct_ids: List[List[Union[int, str]]],
	) -> List[np.ndarray]:
		fused = []
		for id_list in struct_ids:
			masks = self.get_masks(id_list)
			if isinstance(masks, dict):
				masks = functools.reduce(
					np.logical_or, masks.values())
			fused.append(np.array(masks, dtype=bool))
		return fused

	def _setup_meta(self):
		if self.mcc.resolution == 100:
			self.dims = (132, 80, 114)
		elif self.mcc.resolution == 50:
			self.dims = (264, 160, 228)
		elif self.mcc.resolution == 25:
			self.dims = (528, 320, 456)
		elif self.mcc.resolution == 10:
			self.dims = (1320, 800, 1140)
		self.dims_flat = (self.dims[0], self.dims[-1])
		self.origin = (0.0, 0.0, 0.0)
		return

	def _setup_lr(self):
		lr = mask_left_right(np.ones(self.dims), 2)
		self.slices_l = lr['slices_l']
		self.slices_r = lr['slices_r']
		self.mask_l = lr['mask_l']
		self.mask_r = lr['mask_r']
		return

	def make_hemis(self, l_id=1, r_id=2):
		hemis = np.zeros(self.dims)
		hemis[self.slices_l] = l_id
		hemis[self.slices_r] = r_id
		return hemis

	def make_layer_masks(self):
		subtree, _ = self.get_subtree(315, show=False)
		layers = collections.defaultdict(list)
		for node in subtree.leaves():
			tag = node.tag.split('(')[0].strip()
			for ll in ['1', '2/3', '4', '5', '6a', '6b']:
				if tag[-len(ll):] == ll:
					layers[ll].append(node.identifier)
		layers = dict(sorted(layers.items()))
		layer_masks = {}
		for k, struct_ids in layers.items():
			layer_masks[k] = self.get_masks(
				struct_ids, True).astype(bool)
		return layer_masks

	def cortical_map(
			self,
			volume: np.ndarray,
			proj_mode: str = None,
			agg_func: Callable = np.mean,
			fill: Union[int, float] = None, ):

		assert callable(agg_func),\
			f"agg_func must be callable, not {agg_func}"

		if proj_mode is None:
			proj_mode = self.proj_mode
		else:
			self.lookup, self.paths = self.load_paths(proj_mode)

		if self.lookup is None or self.paths is None:
			self.lookup, self.paths = self.load_paths(proj_mode)

		def apply_along_path(i):
			path = self.paths[i]
			path = path[path.nonzero()]
			arr = volume.flat[path]
			return agg_func(arr)

		result = np.zeros(self.lookup.shape, dtype=volume.dtype)
		apply_along_paths_ = np.vectorize(apply_along_path)

		null = -1 if self.proj_mode == 'top_view' else 0
		idxs = np.where(self.lookup > null)
		result[idxs] = apply_along_paths_(self.lookup[idxs])

		if fill is not None:
			result[self.lookup == null] = fill

		return result

	def set_proj(self, proj_mode):
		_allowed = ('top_view', 'dorsal_flatmap')
		assert proj_mode in _allowed,\
			f"allowed projections: '{_allowed}'"
		self.proj_mode = proj_mode
		return

	def load_paths(self, proj_mode=None):
		if proj_mode is not None:
			self.set_proj(proj_mode)
		path = pjoin(self.ws_dir, 'cortical_coordinates/ccf_2017')
		path = pjoin(path, f"{self.proj_mode}_paths_10.h5")
		with h5py.File(path, 'r') as f:
			lookup = f['view lookup'][:]
			paths = f['paths'][:]
		return lookup, paths

	def load_nissl(self, resolution: int = None):
		if resolution is None:
			resolution = self.mcc.resolution
		nissle_file_name = f"ara_nissl_{resolution}.nrrd"
		nissl_file = pjoin(self.ws_dir, nissle_file_name)
		if not os.path.isfile(nissl_file):
			rapi = ReferenceSpaceApi()
			rapi.download_volumetric_data(
				data_path='ara_nissl',
				file_name=nissle_file_name,
				save_file_path=nissl_file,
				voxel_resolution=resolution,
				coordinate_framework='mouse_ccf',
			)
		return nrrd.read(nissl_file)[0]

	def load_info_dfs(self):
		# noinspection PyBroadException
		try:
			set_ids = self.s_tree.get_structure_sets()
			onto_info = OntologiesApi().get_structure_sets(set_ids)
			onto_info = pd.DataFrame(onto_info)
		except Exception:
			onto_info = None
			print('Error loading ontology info')
		expts = self.mcc.get_experiments(dataframe=True)
		return expts, onto_info


def mask_left_right(mask: np.ndarray, lr_axis: int = 2):
	npix_lr = mask.shape[lr_axis]
	coverage = int(np.ceil(npix_lr / 2))
	slices_l = tuple(
		slice(0, coverage) if i == lr_axis
		else slice(mask.shape[i])
		for i in range(len(mask.shape))
	)
	slices_r = tuple(
		slice(npix_lr - coverage, npix_lr)
		if i == lr_axis else slice(mask.shape[i])
		for i in range(len(mask.shape))
	)
	mask_l = mask.copy().astype(bool)
	mask_r = mask.copy().astype(bool)
	mask_l[slices_r] = 0
	mask_r[slices_l] = 0

	lr_results = {
		'slices_l': slices_l,
		'slices_r': slices_r,
		'mask_l': mask_l.astype(bool),
		'mask_r': mask_r.astype(bool),
	}
	return lr_results


def load_bg_atlas(
		atlas: Union[int, str, BrainGlobeAtlas] = 100,
		correct_origin: str = 'asl', ):
	if isinstance(atlas, int):
		atlas = f"allen_mouse_{atlas}um"
		atlas = BrainGlobeAtlas(atlas)
	elif isinstance(atlas, str):
		atlas = BrainGlobeAtlas(atlas)
	tgt_space = AnatomicalSpace(
		origin=correct_origin,
		shape=atlas.space.shape,
		resolution=atlas.space.resolution,
		offset=atlas.space.offset,
	)
	atlas.space = tgt_space
	l, r = atlas.left_hemisphere_value, atlas.right_hemisphere_value
	atlas.left_hemisphere_value, atlas.right_hemisphere_value = r, l
	return atlas


def get_subtree(
		tree: Tree,
		base_id: int = None,
		n_levels: int = None,
		show: bool = True, ):
	base_id = base_id if base_id else tree.root
	base_lvl = tree.level(tree.nodes[base_id].identifier)
	subtree = tree.subtree(base_id)
	n_levels = n_levels if n_levels else subtree.depth()
	if show:
		subtree.show(filter=_get_tree_filter_fn(subtree, n_levels))
	return subtree, base_lvl


def _get_tree_filter_fn(tree, n_levels):
	def _f(x):
		return tree.level(x.identifier) in range(n_levels)
	return _f


def included_intersect_tables(
		s_included: List[str],
		s_main: List[str] = None,
		export_path: str = None,
		allen: Allen = None,
		char_lim: Tuple[int, int] = (np.inf, np.inf),):
	"""
	:param s_included:
	:param s_main:
	:param allen:
	:param char_lim:
	:param export_path:
	:return:
	:example:
		tables = included_intersect_tables(
		s_included=mice.cfg.include_regions,
		export_path=tmp_dir,
		)
	"""

	allen = allen if allen else Allen()
	tree = allen.bg.hierarchy
	s_included = allen.s_tree.get_structures_by_acronym(s_included)
	include_ids = [s['id'] for s in s_included]
	s_main = s_main if s_main else [
		'BS', 'CB', 'CNU',
		'CTXsp', 'HPF', 'OLF',
	]
	s_main = allen.s_tree.get_structures_by_acronym(s_main)

	tables = {}
	for s in s_main:
		subtree, base_lvl = get_subtree(
			tree=tree,
			base_id=s['id'],
			n_levels=3,
			show=False,
		)
		table = get_summary_table(allen, structs=[s['acronym']])

		selected = set(include_ids).intersection(set(subtree.nodes))
		selected = list(selected)
		selected = allen.s_tree.get_structures_by_id(selected)

		rows = collections.defaultdict(list)
		for item in selected:
			lvl = tree.level(item['id'])
			row = [lvl] + _row(allen, item['id'], char_lim)
			rows[lvl].append(row)
		rows = dict(sorted(rows.items()))
		for lvl in rows:
			rows[lvl] = sorted(rows[lvl], key=lambda x: x[3])
		rows = list(itertools.chain(*rows.values()))
		for r in rows:
			table.add_row(r)
		table.add_row(['---'] * len(table.field_names))
		tables[s['acronym']] = table

		if export_path is not None:
			html = table.get_html_string(
				header=True,
				border=True,
				format=True,
			)
			file_name = f"hierarchy_root-{s['acronym']}.html"
			_ = save_obj(
				obj=html,
				file_name=file_name,
				save_dir=export_path,
				mode='html',
			)
	return tables


def get_summary_table(
		allen: Allen = None,
		structs: List[str] = None,
		char_lim: Tuple[int, int] = (33, 10),
		export_path: str = None, ):

	allen = allen if allen else Allen()
	tree = allen.bg.hierarchy
	structs = structs if structs else ['Isocortex', 'MO', 'SS', 'SSp']
	structs = allen.s_tree.get_structures_by_acronym(structs)
	struct_ids = [item['id'] for item in structs]

	field_names = [
		'lvl', 'id', 'acronym (name)',
		'parent', '# vox', 'sz / cortex',
	]
	table = PrettyTable(field_names=field_names)
	table.align['acronym (name)'] = 'l'
	table.align['parent'] = 'l'

	root_lvl = np.inf
	root_struct = structs[0]
	for base_id in struct_ids:
		subtree, base_lvl = get_subtree(
			tree=tree,
			base_id=base_id,
			n_levels=2,
			show=False,
		)
		if base_lvl < root_lvl:
			root_lvl = base_lvl
			root_struct = allen.s_tree.get_structures_by_id([subtree.root])[0]

		table.add_row(['---'] * len(table.field_names))
		for identifier, node in subtree.nodes.items():
			lvl = tree.level(node.identifier)
			if lvl not in [root_lvl, base_lvl + 1]:
				continue
			row = [lvl] + _row(allen, node.identifier, char_lim)
			table.add_row(row)

			if lvl == root_lvl:
				table.add_row(['---'] * len(table.field_names))
	table.add_row(['---'] * len(table.field_names))

	if export_path is not None:
		html = table.get_html_string(header=True, border=True, format=True)
		file_name = f"hierarchy_root-{root_struct['name']}.html"
		_ = save_obj(
			obj=html,
			file_name=file_name,
			save_dir=export_path,
			mode='html',
		)
	return table


def _row(allen, identifier, char_lim: Tuple[int, int]):
	struct = allen.s_tree.get_structures_by_id([identifier])[0]
	mask = allen.mcc.get_structure_mask(struct['id'])[0]
	num_vox = mask.sum()
	relative_sz = num_vox / allen.get_masks('Isocortex').sum() * 100
	relative_sz = f"{int(np.round(relative_sz))} %"

	if '-' in struct['acronym']:
		name = struct['name'].split(',')[1].strip()
	else:
		name = struct['name']
	if len(name) <= char_lim[0]:
		name = f"{struct['acronym']} ({name})"
	else:
		name = f"{struct['acronym']} ({name[:char_lim[0] - 3]}...)"

	parent = allen.s_tree.parent([struct['id']])[0]
	if len(parent['name']) <= char_lim[1]:
		parent = f"{parent['acronym']} ({parent['name']})"
	else:
		parent = f"{parent['acronym']} ({parent['name'][:char_lim[1] - 3]}...)"
	return [
		struct['id'] if struct['id'] < 10000 else '',
		name, parent,
		num_vox, relative_sz,
	]


def make_tree_graph(
		root: str,
		allen: Allen = None,
		n_levels: int = 3,
		use_acroym: bool = False, ):

	allen = allen if allen else Allen()
	lbl_key = 'acronym' if use_acroym else 'name'
	if use_acroym:
		structure = allen.s_tree.get_structures_by_acronym([root])[0]
	else:
		structure = allen.s_tree.get_structures_by_name([root])[0]

	g = nx.DiGraph()
	g.add_node('\n'.join(structure[lbl_key].split(' ')), lvl=0)

	src_ids = [structure['id']]
	dend = {0: src_ids}
	for lvl in range(1, n_levels):
		new_leaves = []
		for i in src_ids:
			g, children = _add_children(
				graph=g,
				src_id=i,
				level=lvl,
				structure_tree=allen.s_tree,
				lbl_key=lbl_key,
			)
			for child in children:
				new_leaves.append(child['id'])
		src_ids = new_leaves
		dend[lvl] = src_ids
	output = {
		'graph': g,
		'dendrogram': dend,
		'node2c': _get_color_dict(g, allen.s_tree, use_acroym),
	}
	return output


def _get_color_dict(graph, structure_tree, use_acroym):
	from utils.plotting import rgb2hex
	node2c = {}
	for i, n in enumerate(graph.nodes(data=True)):
		if use_acroym:
			struct = structure_tree.get_structures_by_acronym([n[0]])[0]
		else:
			fixed_name = n[0].replace('\n', ' ')
			struct = structure_tree.get_structures_by_name([fixed_name])[0]
		node2c[i] = rgb2hex([item / 255 for item in struct['rgb_triplet']])
	return node2c


def _add_children(graph: nx.DiGraph(), src_id, level, structure_tree, lbl_key):
	src = structure_tree.get_structures_by_id([src_id])[0]
	children = structure_tree.children([src_id])[0]
	for child in children:
		src_name = '\n'.join(src[lbl_key].split(' '))
		tgt_name = '\n'.join(child[lbl_key].split(' '))
		graph.add_node(tgt_name, lvl=level)
		graph.add_edge(src_name, tgt_name)
	return graph, children


def _rich_atlas_metadata(atlas_name, metadata):
	orange = "#f59e42"
	dimorange = "#b56510"
	gray = "#A9A9A9"
	mocassin = "#FFE4B5"
	cit_name, cit_link = metadata["citation"].split(", ")

	# Create a rich table
	tb = Table(
		box=None,
		show_lines=False,
		title=atlas_name.replace("_", " ").capitalize(),
		title_style=f"bold {orange}",
	)

	# Add entries to table
	tb.add_column(
		style=f"bold {mocassin}",
		justify="right",
		min_width=8,
		max_width=40,
	)
	tb.add_column(min_width=20, max_width=48)

	tb.add_row(
		"name:",
		Text.from_markup(
			metadata["name"] + f' [{gray}](v{metadata["version"]})'
		),
	)
	tb.add_row("species:", Text.from_markup(f'[i]{metadata["species"]}'))
	tb.add_row("citation:", Text.from_markup(f"{cit_name} [{gray}]{cit_link}"))
	tb.add_row("link:", Text.from_markup(metadata["atlas_link"]))

	tb.add_row("")
	tb.add_row(
		"orientation:",
		Text.from_markup(f"[bold]{metadata['orientation']}"),
	)
	tb.add_row("symmetric:", Pretty(metadata["symmetric"]))
	tb.add_row("resolution:", Pretty(metadata["resolution"]))
	tb.add_row("shape:", Pretty(metadata["shape"]))

	# Fit into panel and yield
	panel = Panel.fit(tb, border_style=dimorange)
	return panel
