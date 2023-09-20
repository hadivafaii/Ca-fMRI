from utils.generic import *


class Clusters(object):
	def __init__(
			self,
			partition: np.ndarray,
			m_struct: Dict[str, np.ndarray],
			c_cluster: Dict[int, str],
			c_struct: Dict[int, str],
			display: str = 'regions',
			threshold: float = 0.0,
			**kwargs,
	):
		super(Clusters, self).__init__()

		idxs = np.unique(partition)
		self.idxs = [int(e) for e in idxs if e > 0]
		self._setup_style(**kwargs)
		self.display = display
		self.set_display(display)

		assert set(self.idxs) == set(c_cluster.keys()),\
			"cluster idxs must match color keys"
		assert m_struct.keys() == c_struct.keys(),\
			"mask & structure color must have same keys"
		self.partition = partition.astype(int)
		self.m_struct = m_struct
		self.c_cluster = c_cluster
		self.c_structure = c_struct

		self._regions_m = {}
		self._clusters_x = {}
		self._clusters_m = {}
		self._get_overlaps()

		self.regions_m = {}
		self.clusters_x = {}
		self.clusters_m = {}
		self.threshold = threshold
		self._apply_threshold()

	def set_threshold(self, new_threshold: float):
		assert 0 <= new_threshold < 100, f"invalid threshold value: {new_threshold}"
		self.threshold = new_threshold
		self._apply_threshold()

	def set_display(self, display: str):
		assert display in ['regions', 'clusters'], f"invalid display option: {display}"
		self.display = display
		self.style['width'] = 55 if self.display == 'clusters' else 45
		self.style['left_col_width'] = 4 if self.display == 'clusters' else 5

	def export(self, export_path: str, name: str = 'overlap'):
		path_regions = save_obj(
			obj=self.repr_html_regions(),
			file_name=f"{name}-regions_n-{len(self.idxs)}.html",
			save_dir=export_path,
			mode='html',
		)
		path_clusters = save_obj(
			obj=self.repr_html_clusters(),
			file_name=f"{name}-clusters_n-{len(self.idxs)}.html",
			save_dir=export_path,
			mode='html',
		)
		return path_regions, path_clusters

	def _setup_style(self, **kwargs):
		style_defaults = {
			'outer_border': 0,
			'inner_border': 1,
			'padding': 10,
			'width': 50,
			'c_fill': 'transparent',
			'left_col_width': 5,
		}
		for k, v in style_defaults.items():
			if k not in kwargs:
				kwargs[k] = v
		self.style = kwargs

	def _get_overlaps(self):
		for acro, m in self.m_struct.items():
			_m = {}
			for i in self.idxs:
				x = self.partition == i
				_m[i] = 100 * np.logical_and(m, x).sum() / m.sum()
			self._regions_m[acro] = _m

		for i in self.idxs:
			x = self.partition == i
			_x = {}
			_m = {}
			for acro, m in self.m_struct.items():
				_x[acro] = 100 * np.logical_and(x, m).sum() / x.sum()
				_m[acro] = 100 * np.logical_and(x, m).sum() / m.sum()
			self._clusters_x[i] = _x
			self._clusters_m[i] = _m

	def _apply_threshold(self):
		# regions
		self.regions_m = {
			k: dict(filter(
				lambda x: int(np.round(x[1])) > 0,
				sorted(v.items(), key=lambda x: x[1], reverse=True)
			)) for k, v in self._regions_m.items()
		}
		# clusters
		self.clusters_x = {
			i: dict(filter(
				lambda x: x[1] > 0.0,
				sorted(v.items(), key=lambda x: x[1], reverse=True)
			)) for i, v in self._clusters_x.items()
		}
		self.clusters_m = {
			i: {k: int(np.round(v)) for k, v in d.items()}
			for i, d in self._clusters_m.items()
		}
		self.clusters_m = {
			i: dict(filter(
				lambda x: x[1] > self.threshold,
				sorted(v.items(), key=lambda x: x[1], reverse=True)
			)) for i, v in self.clusters_m.items()
		}
		return

	def repr_html_clusters(self):
		html = self._init_table()
		for i, c in self.c_cluster.items():
			html += '<tr style="border: 0">'
			html += '<td style="background-color: {:s};'.format(c)
			html += 'border: 0;text-align: center">C_{}</td>'.format(i)
			html += '<td style="border: 0; padding: 10px">'

			html += '<table class="fixed" style="border: 1px solid black;'
			html += 'width: 100%; table-layout: fixed">'

			total_width_covered = 0
			regions_included = []
			for k, overlap in self.clusters_m[i].items():
				try:
					size = self.clusters_x[i][k]
				except KeyError:
					continue
				total_width_covered += size
				regions_included.append(k)

			for k in regions_included:
				size = self.clusters_x[i][k]
				html += '<col width="{}px" />'.format(self.style['width'])
				html += '<col width="{}%" />'.format(size)
			if np.round(total_width_covered) < 99:
				html += '<col width="{}%" />'.format(100 - total_width_covered)

			html += '<tr style="border: 0">'
			for k in regions_included:
				overlap = self.clusters_m[i][k]
				overlap_int = int(np.round(overlap))
				html += '<td style="background-color: transparent;'
				html += 'border: {:d}px solid black;">'.format(self.style['inner_border'])
				html += '<p style="font-family:Courier; text-align: center;">'
				html += '<b>{:s}</b>'.format(k)
				html += '<br>{:d}%</p></td>'.format(overlap_int)

				html += '<td style="'
				html += 'background-color: {:s};">'.format(self.c_structure[k])
				html += '</td>'

			if np.round(total_width_covered) < 99:
				html += '<td style="background-color: {:s};"></td>'.format(self.style['c_fill'])

			html += '</tr></table>'
			html += '</td></tr>'
		html += '</table>'
		return html

	def repr_html_regions(self):
		html = self._init_table()
		for acro, c in self.c_structure.items():
			html += '<tr style="border: 0">'
			html += '<td style="background-color: {:s};'.format(c)
			html += 'border: 0; text-align: center">{}</td>'.format(acro)
			html += '<td style="border: 0; padding: 10px">'

			html += '<table class="fixed" style="border: 1px solid black;'
			html += 'width: 100%; table-layout: fixed">'
			for i, overlap in self.regions_m[acro].items():
				html += '<col width="{}px" />'.format(self.style['width'])
				html += '<col width="{}%" />'.format(overlap)
			html += '<tr style="border: 0">'
			for i, overlap in self.regions_m[acro].items():
				html += '<td style="background-color: transparent;'
				html += 'border: {:d}px solid black;">'.format(self.style['inner_border'])
				html += '<p style="font-family:Courier; text-align: center;">'
				html += '<b>C_{:d}</b>'.format(i)
				html += '<br>{:d}%</p></td>'.format(int(np.round(overlap)))
				html += '<td style="'
				html += 'background-color: {:s};">'.format(self.c_cluster[i])
				html += '</td>'
			html += '</tr></table>'
			html += '</td></tr>'
		html += '</table>'
		return html

	def _init_table(self):
		html = '<table class="fixed" style="border:'
		html += '{}px solid black;'.format(self.style['outer_border'])
		html += 'width: 100%; table-layout: fixed">'
		html += '<col width="{}%" />'.format(self.style['left_col_width'])
		html += '<col width="{}%" />'.format(100 - self.style['left_col_width'])
		return html

	def _repr_html_(self):
		if self.display == 'clusters':
			return self.repr_html_clusters()
		elif self.display == 'regions':
			return self.repr_html_regions()
		else:
			raise RuntimeError()


class ClustersDict(dict):
	def _repr_html_(self):
		html = '<table style="border: 0;">'
		for c in self:
			html += '<tr style="border: 0;">'
			html += '<td style="background-color: {:s}; border: 0;">'.format(c)
			html += '<code style="background-color: {:s};">'.format(c)

			html += c + '</code></td>'
			html += '<td style="border: 0"><code>'
			html += repr(self[c]) + '</code>'
			html += '</td></tr>'
		html += '</table>'

		return html
