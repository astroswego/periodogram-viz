#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from os import makedirs
from os.path import join, isdir
from scipy.signal import lombscargle
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(prog="period_viz")

    parser.add_argument("-i", "--input", type=str,
        help="Input file containing photometry in (time,mag,[err]) columns.")
    parser.add_argument("-o", "--output", type=str,
        help="Directory to output plots.")
    parser.add_argument("-n", "--name", type=str, default="period_viz_",
        help="Name to prefix files with (default 'period_viz_').")
    parser.add_argument("-t", "--type", type=str, default="png",
        help="Image format (default 'png').")
    parser.add_argument("--min-period", type=float, default=0.1,
        help="Minimum period in search space (default 0.1).")
    parser.add_argument("--max-period", type=float, default=10.0,
        help="Maximum period in search space (default 10.0).")
    parser.add_argument("--precision", type=float, default=0.05,
        help="Precision of search space (default 0.05).")

    return parser.parse_args()


def main():
    args = get_args()

    make_sure_path_exists(args.output)

    times, mags, *err = np.loadtxt(args.input, unpack=True)

    periods, pgram = get_pgram(times, mags,
                               args.min_period, args.max_period,
                               args.precision)

    n_periods = len(periods)
    n_digits = int(np.floor(np.log10(n_periods)+1))

    for i, period in enumerate(periods):
        fname = join(args.output,
                     ("{}{:0"+str(n_digits)+"d}.{}").format(args.name, i,
                                                            args.type))

        display(times, mags, periods, pgram, period, fname)


def get_pgram(times, mags, min_period, max_period, precision):
    min_freq, max_freq = np.divide(2*np.pi,
                                   (max_period, min_period))
    freqs = np.arange(min_freq, max_freq, precision)

    scaled_mags = (mags - np.mean(mags)) / np.std(mags)

    pgram = lombscargle(times, scaled_mags, freqs)
    periods = 2*np.pi / freqs

    return np.fliplr(np.vstack((periods, pgram)))


def display(times, mags, periods, pgram, period, fname):
    fig, axes = plt.subplots(1, 2)
    lc_axis, pgram_axis = axes

    lc_axis.invert_yaxis()

    phases = (times / period) % 1.0

    lc_axis.scatter(phases, mags, color="blue", marker=".")

    pgram_axis.plot(periods, pgram, "k-", zorder=2)
    pgram_axis.axvline(period, color="red", linestyle="-", zorder=1)

    fig.savefig(fname)
    plt.close(fig)



def make_sure_path_exists(path):
    try:
        makedirs(path)
    except OSError:
        if not isdir(path):
            raise


if __name__ == "__main__":
    exit(main())
