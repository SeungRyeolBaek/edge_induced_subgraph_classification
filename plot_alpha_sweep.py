# plot_alpha_sweep.py
# Plot mean +/- std of test scores across repeats for alpha sweeps.
#
# Expected CSV format:
#   dataset,alpha,repeat,seed,val,test
#
# Requested changes:
# (1) Put dataset name ABOVE the plot frame (outside axes), not inside.
# (2) Increase ALL font sizes.

import argparse
import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLOR = "#23adcf"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--in_dir", type=str, default="alpha_sweep_results")
    p.add_argument("--datasets", nargs="+", default=["DocRED", "VisualGenome", "Connectome"])
    p.add_argument("--metric", type=str, default="test", choices=["test", "val"])
    p.add_argument("--shade", type=str, default="std", choices=["std", "sem"])

    p.add_argument("--opacity", type=float, default=0.30)
    p.add_argument("--linewidth", type=float, default=3.0)
    p.add_argument("--marker", action="store_true")

    p.add_argument("--xticks", type=str, default="auto")
    p.add_argument("--xlim", type=str, default="auto")
    p.add_argument("--ylim", type=str, default="auto")

    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--save_pdf", action="store_true")
    p.add_argument("--dpi", type=int, default=300)

    # font/size
    p.add_argument("--font_size", type=float, default=20)
    p.add_argument("--figsize", type=str, default="7.2,4.8")

    # dataset name above plot
    p.add_argument(
        "--dataset_title",
        action="store_true",
        help="If set, show dataset name above the axes frame (outside).",
    )
    p.add_argument(
        "--title_pad",
        type=float,
        default=0.02,
        help="Extra vertical space reserved for the dataset title (figure fraction).",
    )

    return p.parse_args()


def _parse_pair(s: str) -> Optional[Tuple[float, float]]:
    s = s.strip()
    if s.lower() == "auto":
        return None
    a, b = s.split(",")
    return float(a), float(b)

def _parse_list(s: str) -> Optional[List[float]]:
    s = s.strip()
    if s.lower() == "auto":
        return None
    if s.lower() == "none":
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_figsize(s: str) -> Tuple[float, float]:
    w, h = s.split(",")
    return float(w), float(h)

def _apply_rcparams(font_size: float):
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size * 1.05,
            "axes.titlesize": font_size * 1.10,
            "xtick.labelsize": font_size * 0.95,
            "ytick.labelsize": font_size * 0.95,
        }
    )

def load_csv(in_dir: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(in_dir, f"{dataset}_alpha_sweep.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")

    df = pd.read_csv(path)
    required = {"dataset", "alpha", "repeat", "seed", "val", "test"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df["alpha"] = df["alpha"].astype(float)
    df["val"] = df["val"].astype(float)
    df["test"] = df["test"].astype(float)
    return df

def summarize(df: pd.DataFrame, metric: str, shade: str) -> pd.DataFrame:
    g = df.groupby("alpha")[metric]
    out = g.agg(["mean", "std", "count"]).reset_index().sort_values("alpha")
    out["sem"] = out["std"] / np.sqrt(np.maximum(out["count"].values, 1))
    band = "std" if shade == "std" else "sem"
    out["band"] = out[band]
    out["low"] = out["mean"] - out["band"]
    out["high"] = out["mean"] + out["band"]
    return out


def plot_one(
    summary_df: pd.DataFrame,
    dataset: str,
    metric: str,
    shade: str,
    opacity: float,
    linewidth: float,
    marker: bool,
    xticks: Optional[List[float]],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    out_path_png: str,
    out_path_pdf: Optional[str],
    dpi: int,
    figsize: Tuple[float, float],
    dataset_title: bool,
    title_pad: float,
):
    x = summary_df["alpha"].values
    y = summary_df["mean"].values
    lo = summary_df["low"].values
    hi = summary_df["high"].values

    fig, ax = plt.subplots(figsize=figsize)

    if marker:
        ax.plot(x, y, color=COLOR, linewidth=linewidth, marker="o")
    else:
        ax.plot(x, y, color=COLOR, linewidth=linewidth)

    ax.fill_between(x, lo, hi, color=COLOR, alpha=opacity, linewidth=0)

    ax.set_xlabel(r"Mixing coefficient $\alpha$")
    ax.set_ylabel(f"Micro-F1")

    if xticks is not None and len(xticks) > 0:
        ax.set_xticks(xticks)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.35)

    # Reserve top margin, then put dataset name ABOVE axes frame.
    # (This is outside the plot box, which is what you want.)
    if dataset_title:
        # make room for suptitle
        fig.tight_layout(rect=[0, 0, 1, 1 - title_pad])
        fig.suptitle(dataset, fontweight="bold", y=1.0)
    else:
        fig.tight_layout()

    fig.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    if out_path_pdf is not None:
        fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    _apply_rcparams(float(args.font_size))
    figsize = _parse_figsize(args.figsize)

    out_dir = args.out_dir if args.out_dir is not None else os.path.join(args.in_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    xlim = _parse_pair(args.xlim)
    ylim = _parse_pair(args.ylim)
    xticks = _parse_list(args.xticks)

    for ds in args.datasets:
        df = load_csv(args.in_dir, ds)
        s = summarize(df, metric=args.metric, shade=args.shade)

        local_xlim = xlim
        if local_xlim is None:
            local_xlim = (float(np.min(s["alpha"])), float(np.max(s["alpha"])))

        png_path = os.path.join(out_dir, f"{ds}_alpha_{args.metric}_mean_pm_{args.shade}.png")
        pdf_path = None
        if args.save_pdf:
            pdf_path = os.path.join(out_dir, f"{ds}_alpha_{args.metric}_mean_pm_{args.shade}.pdf")

        plot_one(
            summary_df=s,
            dataset=ds,
            metric=args.metric,
            shade=args.shade,
            opacity=float(args.opacity),
            linewidth=float(args.linewidth),
            marker=bool(args.marker),
            xticks=xticks,
            xlim=local_xlim,
            ylim=ylim,
            out_path_png=png_path,
            out_path_pdf=pdf_path,
            dpi=int(args.dpi),
            figsize=figsize,
            dataset_title=bool(args.dataset_title),
            title_pad=float(args.title_pad),
        )

        print(f"[saved] {png_path}")
        if pdf_path is not None:
            print(f"[saved] {pdf_path}")


if __name__ == "__main__":
    main()
