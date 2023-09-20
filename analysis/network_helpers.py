from utils.generic import *
import networkx as nx
import networkx.algorithms.community as nx_comm
from nancorrmp.nancorrmp import NaNCorrMp


def make_random_model(
		g: nx.Graph(),
		weighted: bool = False,
		clustered: bool = False,
		random_state: int = 42, ):

	if clustered:
		tri = nx.triangles(g)
		gen_deg = nx.generalized_degree(g)
		joint_seq = [
			(dict(gen_deg[n]).get(0, 0), tri[n])
			for n in g.nodes()
		]
		delta = {}
		g_cfg_all = {}
		for r in range(1, 20 + 1):
			_seq = [(i, j // r) for i, j in joint_seq]
			a, b = tuple(zip(*_seq))
			if sum(a) % 2 != 0 or sum(b) % 3 != 0:
				continue
			g_cfg = nx.random_clustered_graph(_seq, seed=random_state)
			g_cfg = _fix_cfg_graph(g_cfg)
			if weighted:
				g_cfg = _add_weight_cfg(g, g_cfg, random_state)
			g_cfg_all[r] = g_cfg
			delta[r] = _delta_clustering(g, g_cfg, weighted=weighted)

		best_r = list(delta)[np.argmin(list(delta.values()))]
		g_cfg = g_cfg_all[best_r]

	else:
		seq = sorted([d for n, d in g.degree()], reverse=True)
		g_cfg = nx.configuration_model(seq, seed=random_state)
		g_cfg = _fix_cfg_graph(g_cfg)
		if weighted:
			g_cfg = _add_weight_cfg(g, g_cfg, random_state)

	return g_cfg


def _add_weight_cfg(g, g_cfg, random_state):
	rng = get_rng(random_state)
	weights = [d['weight'] for _, _, d in g.edges(data=True)]
	weights = rng.choice(weights, size=len(g_cfg.edges()))
	for w, e in zip(weights, g_cfg.edges(data=True)):
		src, tgt, _ = e
		g_cfg[src][tgt]['weight'] = w
	return g_cfg


def _fix_cfg_graph(g_cfg):
	g_cfg = nx.Graph(g_cfg)
	g_cfg.remove_edges_from(nx.selfloop_edges(g_cfg))
	return g_cfg


def _delta_clustering(g, g_cfg, weighted):
	weight = 'weight' if weighted else None
	c = nx.average_clustering(g, weight=weight)
	c_cfg = nx.average_clustering(g_cfg, weight=weight)
	return abs(c - c_cfg) / c


def save_edgelist(
		graph: nx.Graph,
		file_name: str,
		save_dir: str,
		weighted: bool = False,
		sampler: str = 'binomial',
		sampler_kws: dict = None,
		random_state: int = 42,
		verbose: bool = False, ):

	adj = nx.adjacency_matrix(graph).toarray()
	if weighted:
		rng = get_rng(random_state)
		sampler = getattr(rng, sampler)
		adj_kw = sampler_kws.pop('adj')
		sampler_kws[adj_kw] = adj
		counts = sampler(**sampler_kws)
	else:
		counts = adj.astype(bool).astype(int)

	edges = []
	for i, j in graph.edges(data=False):
		edges.extend([(i, j)] * counts[i, j])
		edges.extend([(j, i)] * counts[i, j])

	lines = [
		"{:d}\t{:d}\n".format(i, j)
		for (i, j) in sorted(edges)
	]
	save_obj(
		obj=lines,
		file_name=file_name,
		save_dir=save_dir,
		mode='txt',
		verbose=verbose,
	)
	return


def get_corrs(
		df: pd.DataFrame,
		percentiles: List[int] = None,
		positive_only: bool = True,
		parallel: bool = True, ):

	if parallel:
		pearson_corr = NaNCorrMp.calculate(df).values
	else:
		pearson_corr = df.corr('pearson').values
	spearman_corr = df.corr('spearman').values

	np.fill_diagonal(pearson_corr, 0.)
	np.fill_diagonal(spearman_corr, 0.)

	if positive_only:
		pearson_corr[pearson_corr < 0.] = 0.
		spearman_corr[spearman_corr < 0.] = 0.

	percentiles = [30, 25, 20, 15, 10] if percentiles is None else percentiles
	percentiles = sorted(percentiles, reverse=True)
	pearson_thresholds = np.percentile(pearson_corr, [100 - p for p in percentiles])
	spearman_thresholds = np.percentile(spearman_corr, [100 - p for p in percentiles])

	pearson, spearman = {}, {}
	for perc, pth, sth in zip(percentiles, pearson_thresholds, spearman_thresholds):
		pmat = pearson_corr.copy()
		smat = spearman_corr.copy()
		pmat[pmat < pth] = 0.
		smat[smat < sth] = 0.
		pearson[perc] = pmat
		spearman[perc] = smat

	output = {
		'percentiles': percentiles,
		'pearson_thresholds': pearson_thresholds,
		'spearman_thresholds': spearman_thresholds,
		'pearson': pearson,
		'spearman': spearman,
		'pearson_corr': pearson_corr,
		'spearman_corr': spearman_corr,
	}
	return output


def convert(
		conver_from: str,
		conver_to: str,
		community: Union[Dict[int, List[int]], List[List[int]]] = None,
		membership: Dict[int, List[int]] = None,
		partition: Dict[int, int] = None,
		random_state: Union[int, np.random.RandomState] = 42, ):

	# TODO: fix membership definition (it should be a vector of probabilities)

	rng = get_rng(random_state)

	if conver_from == 'community':
		community_list = to_list_checker(community)
		num_nodes = max([max(x) for x in community_list]) + 1

		membership = {
			n: list(itertools.compress(
				data=range(len(community_list)),
				selectors=[n in item for item in community_list]
			))
			for n in range(num_nodes)
		}
		partition = {
			n: rng.choice(mem) if len(mem) else -1
			for n, mem in membership.items()
		}

		if conver_to == 'membership':
			return membership
		elif conver_to == 'partition':
			return partition
		else:
			raise ValueError("invalid request: convert_to = {:s}".format(conver_to))

	elif conver_from == 'membership':
		membership_list = to_list_checker(membership)
		num_communities = max([max(x) for x in membership_list]) + 1
		community = {
			i: list(itertools.compress(
				data=range(len(membership_list)),
				selectors=[i in x for x in membership_list]
			))
			for i in range(num_communities)
		}
		partition = {
			n: rng.choice(mem) if len(mem) else -1
			for n, mem in membership.items()
		}

		if conver_to == 'community':
			return community
		elif conver_to == 'partition':
			return partition
		else:
			raise ValueError("invalid request: convert_to = {:s}".format(conver_to))

	elif conver_from == 'partition':
		membership = {n: [i] for n, i in partition.items()}
		membership_list = list(membership.values())
		num_communities = max([max(x) for x in membership_list]) + 1
		community = {
			i: list(itertools.compress(
				data=range(len(membership_list)),
				selectors=[i in x for x in membership_list]
			))
			for i in range(num_communities)
		}
		if conver_to == 'community':
			return community
		elif conver_to == 'membership':
			return membership
		else:
			raise ValueError("invalid request: convert_to = {:s}".format(conver_to))

	else:
		raise ValueError("invalid request: convert_from = {:s}".format(conver_from))


# def run_louvain(
# 		g: nx.Graph(),
# 		res: np.ndarray = None, ):
# 	from cdlib import algorithms as cdlib_algorithms
# 	res = np.logspace(-0.3, 0.3, 1001) if res is None else res
#
# 	# run algorithm
# 	communities = {}
# 	for resolution in tqdm(res, leave=False):
# 		coms = cdlib_algorithms.louvain(
# 			g_original=g,
# 			weight='weight',
# 			resolution=resolution,
# 			randomize=False,
# 		)
# 		communities[resolution] = coms.communities
#
# 	# process results
# 	modularities = {}
# 	num_communities = {}
# 	memberships = {}
# 	partitions = {}
#
# 	for resolution, coms_list in communities.items():
# 		_membership = convert('community', 'membership', community=coms_list)
# 		_partition = convert('community', 'partition', community=coms_list)
# 		# _membership, _partition = get_partition(coms_list, get_rng(random_state))
# 		memberships[resolution] = _membership
# 		partitions[resolution] = _partition
# 		modularities[resolution] = nx_comm.modularity(g, communities=coms_list)
# 		num_communities[resolution] = len(coms_list)
#
# 	results = {
# 		'communities': communities,
# 		'memberships': memberships,
# 		'partitions': partitions,
# 		'modularities': modularities,
# 		'num_communities': num_communities,
# 	}
# 	return results


def to_list_checker(x: Union[dict, list, np.ndarray]):
	if isinstance(x, dict):
		x_list = list(x.values())
	elif isinstance(x, (list, np.ndarray)):
		x_list = list(x)
	else:
		raise ValueError("wrong input type")

	return x_list
