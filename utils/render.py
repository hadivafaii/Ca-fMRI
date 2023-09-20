from .generic import *
from analysis.cluster import Clusters
from register.atlas import load_bg_atlas
from .plotting import complement_color, rgb2hex, sns

from bg_space import AnatomicalSpace

from brainrender import Scene
from brainrender import settings as br_settings
from brainrender.actors.streamlines import make_streamlines
from brainrender.atlas_specific import get_streamlines_for_region

from vedo import Volume as vd_Volume


def cluster2html(
		cluster: Clusters,
		n_cortex_regions: int,
		save_dir: str,
		save_name: str,
		alpha: float = 1.0,
		alpha_root: float = 0.1, ):

	_root_alpha = br_settings.ROOT_ALPHA
	br_settings.ROOT_ALPHA = alpha_root
	br_settings.RENDER = False

	scene = Scene(atlas_name="allen_mouse_100um", root=True, inset=True)
	scene.atlas = load_bg_atlas(scene.atlas)
	src_space = AnatomicalSpace(
		origin='lsa',
		shape=scene.atlas.space.shape,
		resolution=scene.atlas.space.resolution,
		offset=scene.atlas.space.offset,
	)

	surfaces, names = [], []
	for i in cluster.idxs:
		x = cluster.partition == i
		x = src_space.map_stack_to(scene.atlas.space, x)

		vol = vd_Volume(x.astype('uint8'), spacing=scene.atlas.resolution).medianSmooth()
		surf = vol.isosurface().alpha(alpha).lw(0).c(rgb2hex(cluster.c_cluster[i]))
		surf.smoothWSinc().computeNormals().lighting('metallic').phong().decimate().clean()
		surfaces.append(surf)
		names.append('cluster_{:d}'.format(i))
	scene.add(*surfaces, names=names, colors=list(cluster.c_cluster.values()))
	scene.add_silhouette(*scene.actors, lw=0.7, color='k')

	html_file = [
		f"render-{n_cortex_regions}",
		f"n-{len(cluster.idxs)}",
	]
	if save_name:
		html_file = [f"{save_name}"] + html_file
	html_file = '_'.join(html_file) + ".html"
	html_file = pjoin(save_dir, html_file)

	br_settings.RENDER = True
	path = scene.export(html_file)
	br_settings.RENDER = False
	br_settings.ROOT_ALPHA = _root_alpha
	scene.plotter.close()

	return scene, path


def parcel2html(
		masks: dict,
		save_dir: str,
		save_name: str = None,
		cmap='nipy_spectral',
		base_name: str = 'Isocortex',
		c_base: str = '#222f26',
		alpha: float = 0.2,
		alpha_base: float = 0.4,
		alpha_root: float = 0.2, ):

	_root_alpha = br_settings.ROOT_ALPHA
	br_settings.ROOT_ALPHA = alpha_root
	br_settings.RENDER = False

	scene = Scene(atlas_name="allen_mouse_100um", root=True, inset=True)
	scene.atlas = load_bg_atlas(scene.atlas)
	src_space = AnatomicalSpace(
		origin='lsa',
		shape=scene.atlas.space.shape,
		resolution=scene.atlas.space.resolution,
		offset=scene.atlas.space.offset,
	)

	if base_name != 'none':
		scene.add_brain_region(base_name, alpha=alpha_base, color=c_base)
	scene.add_silhouette(*scene.actors, lw=0.7, color='k')

	cmap = sns.color_palette(cmap, as_cmap=True) if isinstance(cmap, str) else cmap
	surfaces, names, colors = [], [], []
	for i, (region_id, x) in enumerate(masks.items()):
		x = src_space.map_stack_to(scene.atlas.space, x)
		c = cmap(i / len(masks))[:3]
		colors.append(c)

		vol = vd_Volume(x.astype('uint8'), spacing=scene.atlas.resolution).medianSmooth()
		surf = vol.isosurface().alpha(alpha).lw(0).c(rgb2hex(c))
		surf.smoothWSinc().computeNormals().lighting('metallic').phong().decimate().clean()
		surfaces.append(surf)
		names.append(f"region_{region_id}")
	scene.add(*surfaces, names=names, colors=colors)

	html_file = [
		f"k{len(masks)}",
	]
	if save_name:
		html_file = [f"{save_name}"] + html_file
	html_file = '_'.join(html_file) + ".html"
	html_file = pjoin(save_dir, html_file)

	br_settings.RENDER = True
	path = scene.export(html_file)
	br_settings.RENDER = False
	br_settings.ROOT_ALPHA = _root_alpha
	scene.plotter.close()

	return scene, path


def tracts2html(
		src_region: str,
		save_dir: str,
		other_regions: List[str] = None,
		alpha_src: float = 0.8,
		alpha_other: float = 0.3,
		alpha_stream: float = 0.15,
		alpha_root: float = 0.2, ):

	_root_alpha = br_settings.ROOT_ALPHA
	br_settings.ROOT_ALPHA = alpha_root
	br_settings.RENDER = False
	scene = Scene(atlas_name="allen_mouse_100um", root=True, inset=True)
	scene.add_brain_region(src_region, alpha=alpha_src)

	if other_regions is not None:
		for region in other_regions:
			scene.add_brain_region(region, alpha=alpha_other)
	scene.add_silhouette(*scene.actors, lw=0.7, color='k')

	# add streams
	color_rgb = scene.get_actors(src_region).pop().color()
	streams_hex = complement_color(*color_rgb)

	streams = get_streamlines_for_region(src_region)
	scene.add(*make_streamlines(*streams, color=streams_hex, alpha=alpha_stream))

	br_settings.RENDER = True
	path = scene.export(pjoin(save_dir, f"streams-{src_region}.html"))
	br_settings.RENDER = False
	br_settings.ROOT_ALPHA = _root_alpha
	scene.plotter.close()

	return scene, path
