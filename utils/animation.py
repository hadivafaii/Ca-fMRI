from .plotting import *
import h5py
import argparse
from matplotlib import animation


def mk_coarse_grained_anim(
        df_all: Dict[str, pd.DataFrame],
        h_load_file: str,
        name: str,
        task: str,
        downsample_sizes: List[int],
        normalize: bool = True,
        fps=5,
        dpi=100,
        save_file=None,):

    # make init fig
    timepoint = 0
    dfs, xy, data_list, vminmax_list, extras = _get_data(df_all, h_load_file, name, task)
    fig, axes, subplots = _mk_fig(
        dfs, xy, data_list, vminmax_list, extras, timepoint, downsample_sizes, normalize, dpi=dpi)
    bar, subplots_arr = subplots

    # downsample all time points
    downsampled = []
    for nbins in downsample_sizes:
        xbins = np.linspace(np.floor(min(xy[:, 0])), np.ceil(max(xy[:, 0])), num=nbins)
        ybins = np.linspace(np.floor(min(xy[:, 1])), np.ceil(max(xy[:, 1])), num=nbins)
        ds = [
            np.concatenate([
                np.expand_dims(downsample(data[t], xy, xbins, ybins, normalize), axis=0) for t in range(extras['nt'])
            ], axis=0)
            for data in data_list
        ]
        downsampled.append(ds)

    # update barplot
    cond = (dfs['performances'].reg_C == extras['best_reg']) & \
           (dfs['performances'].metric.isin(['mcc', 'confidence']))
    df_score = dfs['performances'].loc[cond]

    cond = dfs['coeffs'].reg_C == extras['best_reg']
    df_percent_nonzero = dfs['coeffs'].loc[cond]

    fig_args = {
        'fig': fig,
        'axes': axes,
        'subplots': subplots,
    }
    data_args = {
        'data_list': data_list,
        'downsampled': downsampled,
        'vminmax_list': vminmax_list,
        'extras': extras,
        'df_score': df_score,
        'df_percent_nonzero': df_percent_nonzero,
    }

    ani = animation.FuncAnimation(
        fig=fig,
        func=update_coarse_grained_fig,
        fargs=(fig_args, data_args),
        frames=extras['nt'],
        interval=100,
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={
            'title': "{} / task: {} vs. {}".format(name, extras['pos_lbl'], extras['neg_lbl']),
            'artist': 'Hadi V',
        },
    )
    plt.close(fig)

    if save_file is not None:
        save_dir = os.path.dirname(save_file)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileNotFoundError:
            pass
    else:
        save_file = './animation.mp4'

    ani.save(save_file, writer=writer, dpi=dpi)
    return ani, writer


def update_coarse_grained_fig(step, fig_args, data_args):
    fig = fig_args['fig']
    axes = fig_args['axes']
    subplots = fig_args['subplots']
    bar, subplots_arr = subplots

    data_list = data_args['data_list']
    downsampled = data_args['downsampled']
    vminmax_list = data_args['vminmax_list']
    extras = data_args['extras']
    df_score = data_args['df_score']
    df_percent_nonzero = data_args['df_percent_nonzero']

    # suptitle
    if step == extras['best_timepoint']:
        fig.suptitle("t = {:d} (best timepoint)".format(step), fontsize=20)
    else:
        fig.suptitle("t = {:d}".format(step), fontsize=20)

    # bar plot
    scores = df_score.loc[df_score.timepoint == step].score.to_numpy()
    scores = scores.reshape(extras['nb_seeds'], 2)
    scores_mean = scores.mean(0)
    for patch, ss in zip(bar, scores_mean):
        patch.set_height(ss)

    # scatter
    for i in range(3):
        data = data_list[i][step]
        f = data / vminmax_list[i]
        color_indxs = np.floor(128 * (1 + f))
        color_indxs = np.array(color_indxs, dtype=int)
        if i == 0:
            rgba = cm.seismic(color_indxs)
        else:
            rgba = cm.PiYG_r(color_indxs)

        subplots_arr[i, 0].set_color(rgba)
        subplots_arr[i, 0].set_edgecolor('k')

        # iamge
        for j in range(1, len(downsampled) + 1):
            subplots_arr[i, j].set_data(downsampled[j - 1][i][step])

    # y label
    percent_nonzero = df_percent_nonzero.loc[df_percent_nonzero.timepoint == step].percent_nonzero.to_numpy()
    percent_nonzero = percent_nonzero.reshape(extras['nb_seeds'], extras['nc'])
    y_lbl = "coeffs,  {:s} nonzero: {:.2f} ± {:.2f}".format('%', percent_nonzero.mean(), percent_nonzero.std())
    axes[1][0, -1].set_ylabel(y_lbl)


def mk_coarse_grained_plot(
        df_all: Dict[str, pd.DataFrame],
        h_load_file: str,
        name: str,
        task: str,
        timepoint: int,
        downsample_sizes: List[int] = None,
        normalize: bool = True,
        save_file=None,
        display=True,
        figsize=(24, 13),
        dpi=300,
):
    dfs, xy, data_list, vminmax_list, extras = _get_data(df_all, h_load_file, name, task)
    fig, axes, subplots = _mk_fig(
        dfs, xy, data_list, vminmax_list, extras, timepoint, downsample_sizes, normalize, figsize, dpi)

    msg = "t = {:d}, task = {:s} vs. {:s}"
    msg = msg.format(timepoint, extras['pos_lbl'], extras['neg_lbl'])
    sup = fig.suptitle(msg, fontsize=20)
    save_fig(fig, sup, save_file, display)

    return fig, sup, axes, subplots


def _mk_fig(
        dfs: Dict[str, pd.DataFrame],
        xy: np.ndarray,
        data_list: List[np.ndarray],
        vminmax_list: List[int],
        extras: dict,
        timepoint: int,
        downsample_sizes: List[int] = None,
        normalize: bool = True,
        figsize=(24, 13),
        dpi=200,
        tight_layout: bool = False,
):
    if downsample_sizes is None:
        downsample_sizes = [16, 8, 4, 2]

    # get the data at certain timepoint
    datalist = [item[timepoint] for item in data_list]

    sns.set_style('white')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(nrows=3, ncols=2 + len(downsample_sizes),
                  width_ratios=[0.2, 1] + [0.8] * len(downsample_sizes))

    cond = (dfs['performances'].timepoint == timepoint) & \
           (dfs['performances'].reg_C == extras['best_reg']) & \
           (dfs['performances'].metric.isin(['mcc', 'confidence']))
    scores = dfs['performances'].loc[cond].score.to_numpy()
    scores = scores.reshape(extras['nb_seeds'], 2)

    ax0 = fig.add_subplot(gs[:, 0])
    bar = ax0.bar(
        x=[0, 1],
        height=scores.mean(0),
        color=['gold', 'indigo'],
    )
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(['mcc', 'confidence'])
    ax0.tick_params(axis='x', labelrotation=90)
    ax0.set_xlabel('')
    ax0.set_yticks(np.linspace(0, 1, num=11))

    threshold = 0.9
    ax0.axhspan(ymin=extras['max_score'] * threshold, ymax=extras['max_score'],
                color='deepskyblue', alpha=0.2, zorder=0, label='max score area')
    ax0.axhline(xmin=0, y=extras['max_score'] * threshold, xmax=1, color='navy', ls=':', alpha=0.3, zorder=10)
    ax0.axhline(xmin=0, y=extras['max_score'] - 0.001, xmax=1, color='navy', ls=':', alpha=0.3, zorder=10)
    ax0.set_ylim(0, 1)

    ax0.legend(loc=(0, 1.01))

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title('full', fontsize=15)
    s1 = ax1.scatter(
        x=xy[:, 0],
        y=xy[:, 1],
        c=datalist[0],
        cmap='seismic',
        s=100,
        vmin=-vminmax_list[0],
        vmax=vminmax_list[0],
        edgecolors='k',
        linewidths=0.4,
    )
    plt.colorbar(s1, ax=ax1)

    ax2 = fig.add_subplot(gs[1, 1])
    s2 = ax2.scatter(
        x=xy[:, 0],
        y=xy[:, 1],
        c=datalist[1],
        cmap='PiYG_r',
        s=100,
        vmin=-vminmax_list[1],
        vmax=vminmax_list[1],
        edgecolors='k',
        linewidths=0.4,
    )
    plt.colorbar(s2, ax=ax2)

    ax3 = fig.add_subplot(gs[2, 1])
    s3 = ax3.scatter(
        x=xy[:, 0],
        y=xy[:, 1],
        c=datalist[2],
        cmap='PiYG_r',
        s=100,
        vmin=-vminmax_list[2],
        vmax=vminmax_list[2],
        edgecolors='k',
        linewidths=0.4,
    )
    plt.colorbar(s3, ax=ax3)

    # downsample part
    ax_all = [[ax1, ax2, ax3]]
    subplots_all = [[s1, s2, s3]]
    for col, nbins in enumerate(downsample_sizes):
        gs_list = [gs[i, 2 + col] for i in range(3)]
        fig, ax_list, subplot_list = _mk_downsampled(datalist, vminmax_list, xy, fig,
                                                     gs_list=gs_list, nbins=nbins, normalize=normalize)
        ax_all.append(ax_list)
        subplots_all.append(subplot_list)

    ax_all = np.array(ax_all).T
    subplots_all = np.array(subplots_all).T

    # make y lbls
    cond = (dfs['coeffs'].reg_C == extras['best_reg']) & (dfs['coeffs'].timepoint == timepoint)
    percent_nonzero = dfs['coeffs'].loc[cond].percent_nonzero.to_numpy()
    percent_nonzero = percent_nonzero.reshape(extras['nb_seeds'], extras['nc'])
    percent_nonzero = np.unique(percent_nonzero, axis=1)
    y_lbls = [
        "coeffs,  {:s} nonzero: {:.2f} ± {:.2f}".format('%', percent_nonzero.mean(), percent_nonzero.std()),
        "avg dff ({}),  num = {:d}".format(extras['pos_lbl'], extras['num_pos']),
        "avg dff ({}),  num = {:d}".format(extras['neg_lbl'], extras['num_neg']),
    ]
    for _ax, lbl in zip(ax_all[:, -1], y_lbls):
        _ax.set_ylabel(lbl, rotation=270, fontsize=15, labelpad=20)
        _ax.yaxis.set_label_position("right")

    if tight_layout:
        fig.tight_layout()
    return fig, [ax0, ax_all], [bar, subplots_all]


def _mk_downsampled(datalist, vminmax_list, xy, fig, gs_list, nbins, normalize):
    xbins = np.linspace(np.floor(min(xy[:, 0])), np.ceil(max(xy[:, 0])), num=nbins)
    ybins = np.linspace(np.floor(min(xy[:, 1])), np.ceil(max(xy[:, 1])), num=nbins)
    data_ds = [downsample(data, xy, xbins, ybins, normalize) for data in datalist]

    ax_list = []
    subplot_list = []
    for i in range(3):
        ax = fig.add_subplot(gs_list[i])
        if i == 0:
            ax.set_title('downsample {} x {}'.format(nbins, nbins), fontsize=15)
            _cmap = 'seismic'
        else:
            _cmap = 'PiYG_r'

        im = ax.imshow(data_ds[i], cmap=_cmap, vmin=-vminmax_list[i], vmax=vminmax_list[i])
        ax.set_xticks(range(nbins))
        ax.set_xticklabels([int(e) for e in xbins])
        ax.set_yticks(range(nbins))
        ax.set_yticklabels([int(e) for e in ybins])
        ax.tick_params(axis='x', labelrotation=90)
        ax.invert_yaxis()
        ax.grid()
        if i == 0:
            ax.set_title('downsample {} x {}'.format(nbins, nbins), fontsize=15)
        ax_list.append(ax)
        subplot_list.append(im)

    return fig, ax_list, subplot_list


def _get_data(df_all, h_load_file, name, task):
    # get dfs
    dfs = {k: v.loc[(v.name == name) & (v.task == task)] for k, v in df_all.items()}

    max_score = dfs['performances_filtered'].loc[dfs['performances_filtered'].metric == 'mcc'].score.mean()
    best_reg = dfs['performances_filtered'].best_reg.unique().item()
    best_timepoint = dfs['performances_filtered'].best_timepoint.unique().item()

    nb_seeds = len(dfs['coeffs'].seed.unique())
    nc = len(dfs['coeffs'].cell_indx.unique())
    nt = len(dfs['coeffs'].timepoint.unique())

    z_all = dfs['coeffs'].loc[dfs['coeffs'].reg_C == best_reg].coeffs.to_numpy()
    z_all = z_all.reshape(nb_seeds, nt, nc).mean(0)
    vminmax = np.max(np.abs(z_all))

    cond = dfs['coeffs'].reg_C == best_reg
    z = dfs['coeffs'].loc[cond].coeffs.to_numpy()
    z = z.reshape(nb_seeds, nt, -1).mean(0)

    # get DFFs and XY
    h5_file = h5py.File(h_load_file, "r")
    behavior = h5_file[name]["behavior"]
    trial_info_grp = behavior["trial_info"]

    good_cells = np.array(behavior["good_cells"], dtype=int)
    xy = np.array(behavior["xy"], dtype=float)[good_cells]
    dff = np.array(behavior["dff"], dtype=float)[..., good_cells]

    trial_info = {}
    for k, v in trial_info_grp.items():
        trial_info[k] = np.array(v, dtype=int)
    h5_file.close()

    # dff data
    pos_lbl, neg_lbl = task.split('/')
    pos = trial_info[pos_lbl]
    neg = trial_info[neg_lbl]

    num_pos = sum(pos == 1)
    num_neg = sum(neg == 1)

    dff_pos = dff[:, pos == 1, :].mean(1)
    dff_neg = dff[:, neg == 1, :].mean(1)
    vminmax_pos_dff = np.max(np.abs(dff_pos))
    vminmax_neg_dff = np.max(np.abs(dff_neg))

    data_list = [z, dff_pos, dff_neg]
    vminmax_list = [vminmax, vminmax_pos_dff, vminmax_neg_dff]

    extras = {
        "pos_lbl": pos_lbl,
        "neg_lbl": neg_lbl,
        "num_pos": num_pos,
        "num_neg": num_neg,
        "nc": nc,
        "nt": nt,
        "nb_seeds": nb_seeds,
        "max_score": max_score,
        "best_reg": best_reg,
        "best_timepoint": best_timepoint,
    }

    return dfs, xy, data_list, vminmax_list, extras


def _setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cm",
        help="comment describing the fit to use",
        type=str,
    )
    parser.add_argument(
        "--clf_type",
        help="classifier type, choices: {'logreg', 'svm'}",
        type=str,
        choices={'logreg', 'svm'},
        default='logreg',
    )
    parser.add_argument(
        "--fps",
        help="frames per second",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dpi",
        help="dots per inch",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--normalize",
        help="if true, downsampled data will be normalized",
        action="store_true",
    )
    parser.add_argument(
        "--nb_std",
        help="outlier removal threshold",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--verbose",
        help="verbosity",
        action="store_true",
    )
    parser.add_argument(
        "--base_dir",
        help="base dir where project is saved",
        type=str,
        default='Documents/PROJECTS/Kanold',
    )

    return parser.parse_args()


def main():
    args = _setup_args()

    base_dir = pjoin(os.environ['HOME'], args.base_dir)
    processed_dir = pjoin(base_dir, 'python_processed')
    results_dir = pjoin(base_dir, 'results', args.clf_type, args.cm)
    h_load_file = pjoin(processed_dir, "organized_nb_std={:d}.h5".format(args.nb_std))

    if args.verbose:
        print("[INFO] files found:\n{}".format(os.listdir(results_dir)))
    df_all = {}     # load_dfs(results_dir)

    names = df_all['performances_filtered'].name.unique().tolist()
    tasks = []  # get_tasks()
    downsample_sizes = [16, 8, 4, 2]

    for name in tqdm(names, disable=not args.verbose, dynamic_ncols=True):
        save_dir = pjoin(results_dir, 'individual_results', name)
        os.makedirs(save_dir, exist_ok=True)
        for task in tqdm(tasks, disable=not args.verbose, leave=False):
            cond = (df_all['performances_filtered'].name == name) & (df_all['performances_filtered'].task == task)
            if not sum(cond):
                continue
            pos_lbl, neg_lbl = task.split('/')
            save_file = pjoin(save_dir, '{:s}-{:s}.mp4'.format(pos_lbl, neg_lbl))
            _ = mk_coarse_grained_anim(
                df_all,
                h_load_file,
                name,
                task,
                downsample_sizes,
                normalize=args.normalize,
                fps=args.fps,
                dpi=args.dpi,
                save_file=save_file,
            )

    print("[PROGRESS] done.\n")


if __name__ == "__main__":
    main()
