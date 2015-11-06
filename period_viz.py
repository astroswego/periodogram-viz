#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from os import makedirs
from os.path import join, isdir
from scipy.signal import lombscargle
from argparse import ArgumentParser
from matplotlib import animation


def get_args():
    parser = ArgumentParser(prog="period_viz")

    parser.add_argument("-i", "--input", type=str,
        help="Input file containing photometry in (time,mag,[err]) columns.")
    parser.add_argument("-o", "--output", type=str,
        help="Directory to output plots.")
    parser.add_argument("-n", "--name", type=str, default="period_viz_",
        help="Name to prefix files with (default 'period_viz_').")
    parser.add_argument("-t", "--type", type=str, default="png",
        help="Image format. If format is 'gif', outputs a single gif loop. "
             "Otherwise outputs a series of numbered images (default 'png').")
    parser.add_argument("-f", "--fps", type=int, default=30,
        help="Frames per second if type is 'gif' (default 30).")
    parser.add_argument("-p", "--period", type=float, default=None,
        help="The 'true' period. This will be zoomed in on (default None).")
    parser.add_argument("--min-period", type=float, default=0.1,
        help="Minimum period in search space (default 0.1).")
    parser.add_argument("--max-period", type=float, default=10.0,
        help="Maximum period in search space (default 10.0).")
    parser.add_argument("--coarse-precision", type=float, default=0.05,
        help="Coarse precision of search space (default 0.05).")
    parser.add_argument("--fine-precision", type=float, default=0.01,
        help="Fine precision of search space near true period (default 0.01).")
    parser.add_argument("--fine-radius", type=float, default=0.1,
        help="Radius to cover with fine precision (default 0.1).")

    return parser.parse_args()


def main():
    args = get_args()

    make_sure_path_exists(args.output)

    fig, axes = plt.subplots(1, 2)

    times, mags, *err = np.loadtxt(args.input, unpack=True)

    periods = get_periods(args.period,
                          args.min_period, args.max_period,
                          args.coarse_precision, args.fine_precision,
                          args.fine_radius)
    pgram = get_pgram(times, mags, periods)

    n_periods = len(periods)
    n_digits = int(np.floor(np.log10(n_periods)+1))

    if args.type == "gif":
        def animate_i(i):
            return animate(fig, times, mags, periods, pgram, periods[i])

        fname = join(args.output, args.name+".gif")
        anim = animation.FuncAnimation(fig, animate_i, frames=n_periods)
        anim.save(fname, writer="imagemagick", fps=args.fps)
    else:
        for i, period in enumerate(periods):
            animate(fig, times, mags, periods, pgram, period)

            fname = join(args.output,
                         ("{}{:0"+str(n_digits)+"d}.{}").format(args.name, i,
                                                                args.type))
            fig.savefig(fname)


def get_periods(period,
                min_period, max_period,
                coarse_precision, fine_precision,
                fine_radius):
    if period is None:
        periods = np.arange(min_period, max_period, coarse_precision)
    else:
        radius_low, radius_high = period + np.multiply([-1, +1], fine_radius)

        periods_low = np.arange(min_period, radius_low, coarse_precision)
        periods_mid = np.arange(radius_low, radius_high, fine_precision)
        periods_high = np.arange(radius_high, max_period, coarse_precision)

        periods = np.concatenate((periods_low, periods_mid, periods_high))

    return periods


def get_pgram(times, mags, periods):
    freqs = 2*np.pi / periods
    scaled_mags = (mags - np.mean(mags)) / np.std(mags)

    pgram = lombscargle(times, scaled_mags, freqs)

    return pgram


def animate(fig, times, mags, periods, pgram, period):
    lc_axis, pgram_axis = fig.get_axes()
    lc_axis.clear()
    pgram_axis.clear()

    lc_axis.invert_yaxis()

    phases = (times / period) % 1.0

    lc_axis.scatter(phases, mags, color="blue", marker=".")

    pgram_axis.plot(periods, pgram, "k-", zorder=2)
    pgram_axis.axvline(period, color="red", linestyle="-", zorder=1)


def make_sure_path_exists(path):
    try:
        makedirs(path)
    except OSError:
        if not isdir(path):
            raise


if __name__ == "__main__":
    exit(main())
