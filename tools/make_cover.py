#!/usr/bin/env python3
"""Cover images for the article and the GitHub social preview.

The picture is real data from kvcalc: one 80 GiB GPU serving Llama-3.1-70B at TP=2.
Each square is 1 GiB: weights, engine overhead (assumed), KV cache, and the 10% the engine leaves unclaimed.
Usage: python3 tools/make_cover.py [outdir]   (default docs/img)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvcalc  # noqa: E402

BG, INK, INK2, MUTED = "#16181f", "#f0efec", "#c3c2b7", "#898781"
BLUE, ORANGE, GREY, EDGE = "#3987e5", "#eb6834", "#6b6a66", "#44443f"
SIZES = {"cover-medium-2200x1000.png": (2200, 1000),
         "cover-16x9-1920x1080.png": (1920, 1080),
         "github-social-1280x640.png": (1280, 640)}


def numbers():
    p = kvcalc.plan("Llama-3.1-70B", 2, 80, 0.90, 2, 2, 3.0, 8192)
    weights = round(p["weights_gib"] / 2)          # per GPU
    overhead = 3
    kv = round(p["budget_gib"])
    reserved = round(80 * (1 - 0.90))
    assert weights + overhead + kv + reserved == 80, (weights, overhead, kv, reserved)
    return weights, overhead, kv, reserved, p["conc"]


def draw(path, w, h):
    weights, overhead, kv, reserved, conc = numbers()
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    pt = lambda px: px * 0.72  # px -> points at dpi 100

    # GPU memory grid: 8 columns x 10 rows, 1 GiB per cell, right-aligned
    step = 0.080 * h
    cell = 0.068 * h
    x0 = w - 0.07 * h - 8 * step
    y0 = 0.10 * h
    kinds = ["w"] * weights + ["o"] * overhead + ["k"] * kv + ["r"] * reserved
    colors = {"w": BLUE, "o": GREY, "k": ORANGE}
    for i, kind in enumerate(kinds):
        r, c = divmod(i, 8)
        x, y = x0 + c * step, y0 + r * step
        if kind == "r":
            ax.add_patch(FancyBboxPatch((x, y), cell, cell, boxstyle="round,pad=0,rounding_size=%f" % (0.012 * h),
                                        fc="none", ec=EDGE, lw=max(1.2, h / 500)))
        else:
            ax.add_patch(FancyBboxPatch((x, y), cell, cell, boxstyle="round,pad=0,rounding_size=%f" % (0.012 * h),
                                        fc=colors[kind], ec="none"))

    # text column
    lx = 0.055 * h * 1.25
    ax.text(lx, 0.17 * h, "LLM INFERENCE ON KUBERNETES", color=ORANGE, fontsize=pt(0.036 * h), fontweight="bold", va="center")
    ax.text(lx, 0.36 * h, "Your GPU is", color=INK, fontsize=pt(0.125 * h), fontweight="bold", va="center")
    ax.text(lx, 0.52 * h, "not a counter.", color=INK, fontsize=pt(0.125 * h), fontweight="bold", va="center")
    ax.text(lx, 0.645 * h, "DRA  ·  KV-cache math  ·  routing  ·  autoscaling", color=INK2, fontsize=pt(0.037 * h), va="center")
    ax.text(lx, 0.765 * h, f"{kv} GiB left for KV cache = {conc:.1f} concurrent 8k sequences", color=ORANGE,
            fontsize=pt(0.036 * h), fontweight="bold", va="center")
    legend = [(BLUE, f"weights {weights}"), (GREY, f"overhead {overhead}"), (ORANGE, f"KV cache {kv}"), (None, f"unclaimed {reserved}")]
    x = lx
    for color, label in legend:
        s = 0.030 * h
        y = 0.875 * h
        if color:
            ax.add_patch(FancyBboxPatch((x, y - s / 2), s, s, boxstyle="round,pad=0,rounding_size=%f" % (0.006 * h), fc=color, ec="none"))
        else:
            ax.add_patch(FancyBboxPatch((x, y - s / 2), s, s, boxstyle="round,pad=0,rounding_size=%f" % (0.006 * h), fc="none", ec=EDGE, lw=1.4))
        ax.text(x + s * 1.35, y, label, color=INK2, fontsize=pt(0.030 * h), va="center")
        x += 0.285 * h if w / h > 2 else 0.235 * h
    ax.text(lx, 0.94 * h, "One 80 GiB GPU · Llama-3.1-70B · TP=2 · 1 square = 1 GiB", color=MUTED, fontsize=pt(0.026 * h), va="center")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    print("wrote", path, f"{w}x{h}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img")
    os.makedirs(out, exist_ok=True)
    for name, (w, h) in SIZES.items():
        draw(os.path.join(out, name), w, h)
