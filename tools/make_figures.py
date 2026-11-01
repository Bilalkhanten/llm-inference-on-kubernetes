#!/usr/bin/env python3
"""Draw the article figures from the repo's own tools. Nothing here is typed by hand:
every number comes from kvcalc / captool / routesim, so `make figures` always matches the code.

Output: docs/img/fig1..fig5 (PNG). Palette validated with the dataviz skill's validator:
blue / orange / aqua on a light surface; aqua is below 3:1 so every aqua mark is direct-labelled.
"""
import math
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import captool  # noqa: E402
import kvcalc  # noqa: E402
import routesim  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "img")
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
})


def title(fig, main, sub):
    fig.text(0.015, 0.975, main, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.925, sub, fontsize=11.5, color=INK2, va="top")


def clean(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(length=0)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


# ---------------------------------------------------------------- data ----
KV_CONFIGS = [("Mistral-7B-v0.3", 1), ("Llama-3.1-8B", 1), ("Qwen2.5-32B", 1), ("Qwen2.5-32B", 2),
              ("Llama-3.1-70B", 1), ("Llama-3.1-70B", 2), ("Llama-3.1-70B", 4), ("Llama-3.1-405B", 8)]


def kv_data(gpu_gib=80, util=0.90, overhead=3.0, ctx=8192):
    rows = []
    for name, tp in KV_CONFIGS:
        b = kvcalc.plan(name, tp, gpu_gib, util, 2, 2, overhead, ctx)
        f = kvcalc.plan(name, tp, gpu_gib, util, 2, 1, overhead, ctx)
        rows.append((name, tp, b["conc"] if b else None, f["conc"] if f else None))
    return rows


def route_data(lam=80.0, zipf=1.0, seeds=range(5), stale=0.25, policies=None):
    policies = policies or ["random", "round-robin", "least-queue", "prefix-hash", "scored"]
    out = {}
    for pol in policies:
        res = [routesim.Sim(pol, s, lam=lam, zipf=zipf, stale=stale).go() for s in seeds]
        out[pol] = {"p50": st.mean(routesim.pct(r["ttft"], 50) for r in res),
                    "p99": st.mean(routesim.pct(r["ttft"], 99) for r in res),
                    "hit": st.mean(r["hit"] for r in res)}
    return out


# ------------------------------------------------------------- figures ----
def fig_architecture():
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.86)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 74)
    ax.axis("off")
    title(fig, "Four layers, three loops", "Where each piece of the stack plugs into the request path")

    def box(x, y, w, h, text, color, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    fc=SURFACE, ec=color, lw=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11,
                color=INK, fontweight="bold" if bold else "normal", linespacing=1.3)

    def arrow(p, q, color, label=None, lpos=None, style="-|>", ls="-"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=color, lw=2, ls=ls,
                                                        shrinkA=0, shrinkB=0))
        if label:
            ax.text(*lpos, label, fontsize=9.5, color=INK2, ha="center", va="center")

    # request path (blue)
    box(1, 40, 13, 12, "Client\n/v1/chat/\ncompletions", BLUE)
    box(21, 40, 18, 12, "Gateway\n+ HTTPRoute", BLUE)
    box(46, 40, 20, 12, "Endpoint Picker\n(EPP, ext-proc)", BLUE)
    box(73, 36, 26, 20, "vLLM replicas\none GPU claim each\nKV cache + queue", BLUE, bold=True)
    arrow((14, 46), (21, 46), BLUE)
    arrow((39, 46), (46, 46), BLUE)
    arrow((66, 46), (73, 46), BLUE, "pick", (69.5, 48.8))
    arrow((73, 41), (66, 41), BLUE, style="-|>", ls="--")
    ax.text(68.2, 39.3, "metrics", fontsize=9.5, color=INK2, ha="center", va="top")
    # device allocation (aqua)
    box(40, 62, 24, 10, "DRA driver\npublishes ResourceSlices", AQUA)
    box(70, 62, 29, 10, "Scheduler allocates\nResourceClaim (CEL)", AQUA)
    arrow((64, 67), (70, 67), AQUA)
    arrow((86, 62), (86, 56), AQUA, "GPU(s) bound", (92.5, 59))
    # scaling loop (orange)
    box(73, 17, 26, 11, "Prometheus", ORANGE)
    box(40, 17, 24, 11, "KEDA / HPA\nqueue + KV usage", ORANGE)
    arrow((86, 36), (86, 28), ORANGE, "scrape /metrics", (93.5, 32))
    arrow((73, 22.5), (64, 22.5), ORANGE, "query", (68.5, 25.5))
    arrow((62, 28), (79, 36), ORANGE, "scale replicas\n(cold start: minutes)", (49, 34.5), ls="-")
    # legend
    for i, (c, t) in enumerate([(BLUE, "request path"), (AQUA, "device allocation"), (ORANGE, "scaling loop")]):
        x = 4 + i * 33
        ax.plot([x, x + 6], [6, 6], color=c, lw=3)
        ax.text(x + 8, 6, t, fontsize=11, color=INK, va="center")
    save(fig, "fig1-architecture.png")


def fig_kv(rows):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    fig.subplots_adjust(0.25, 0.10, 0.97, 0.84)
    title(fig, "How many 8k-token sequences fit per replica?", "80 GiB GPUs, 90% memory utilisation, 3 GiB/GPU overhead (assumed), bf16 weights")
    n = len(rows)
    for i, (name, tp, bf, f8) in enumerate(rows):
        y = n - 1 - i
        label = f"{name.replace('-v0.3', '')}  TP={tp}"
        if bf is None:
            ax.text(1, y, "does not fit (weights exceed the GPU budget)", fontsize=10.5, color=INK2, va="center", style="italic")
        else:
            ax.barh(y + 0.19, bf, 0.34, color=BLUE, ec=SURFACE, lw=1.5)
            ax.barh(y - 0.19, f8, 0.34, color=ORANGE, ec=SURFACE, lw=1.5)
            ax.text(bf + 1.5, y + 0.19, f"{bf:.1f}", va="center", fontsize=10.5, color=INK)
            ax.text(f8 + 1.5, y - 0.19, f"{f8:.1f}", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{r[0].replace('-v0.3', '')}  TP={r[1]}" for r in rows][::-1], fontsize=11)
    ax.set_xlim(0, 135)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("concurrent sequences at 8k average context")
    clean(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(h, ["bf16 KV cache", "fp8 KV cache"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=11)
    save(fig, "fig2-kv-capacity.png")


def fig_routing(a, b, clip=10.0):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.4))
    fig.subplots_adjust(0.17, 0.07, 0.97, 0.84, hspace=0.42)
    title(fig, "Time to first token by routing policy", "8 replicas, 80 req/s. Bars are capped at 10 s; off-scale values are written out")
    order = ["random", "round-robin", "least-queue", "prefix-hash", "scored"]
    for ax, data, label in ((axes[0], a, "A) mild prefix skew (Zipf 1.0)"), (axes[1], b, "B) hot-prefix skew (Zipf 1.6)")):
        for i, pol in enumerate(order):
            y = len(order) - 1 - i
            for off, key, color in ((0.19, "p50", BLUE), (-0.19, "p99", ORANGE)):
                v = data[pol][key]
                ax.barh(y + off, min(v, clip), 0.34, color=color, ec=SURFACE, lw=1.5)
                txt = f"{v:.2f} s" if v < 100 else f"{v:.0f} s"
                if v > clip:
                    txt += "  (off scale)"
                ax.text(min(v, clip) + 0.15, y + off, txt, va="center", fontsize=10, color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order[::-1], fontsize=11)
        ax.set_xlim(0, clip * 1.42)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_title(label, loc="left", fontsize=12, color=INK, fontweight="bold", pad=8)
        clean(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    axes[0].legend(h, ["p50 TTFT", "p99 TTFT"], loc="lower right", frameon=False, fontsize=11)
    axes[1].set_xlabel("seconds")
    save(fig, "fig3-routing-latency.png")


def fig_staleness(staleness, series):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    fig.subplots_adjust(0.10, 0.14, 0.82, 0.84)
    title(fig, "Stale metrics turn least-queue into a herd", "p50 TTFT at 80 req/s as the router's view of queue depth gets older")
    xs = list(range(len(staleness)))
    styles = {"round-robin": (INK2, "--", "round-robin"), "least-queue": (BLUE, "-", "least-queue"), "scored": (ORANGE, "-", "scored")}
    for pol, vals in series.items():
        color, ls, lab = styles[pol]
        ax.plot(xs, vals, color=color, lw=2.2, ls=ls, marker="o", ms=7, mec=SURFACE, mew=2)
        ax.text(xs[-1] + 0.12, vals[-1], f"{lab}\n{vals[-1]:.2f} s", color=INK, fontsize=10.5, va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(s * 1000)} ms" if s < 1 else f"{s:g} s" for s in staleness])
    ax.set_xlim(-0.2, len(xs) - 1 + 0.15)
    ax.set_xlabel("age of the queue-depth snapshot the router sees")
    ax.set_ylabel("p50 TTFT (s)")
    clean(ax, "y")
    save(fig, "fig4-staleness.png")


def fig_headroom():
    _, usable, base = captool.steady_replicas(60, 6.0, 54, 0.70)
    ms = list(range(0, 26))
    vals = [captool.headroom_replicas(20, m, 6.0, usable)[2] if m else 0 for m in ms]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    fig.subplots_adjust(0.10, 0.14, 0.96, 0.84)
    title(fig, "The cold start sets your floor", "Extra replicas to pre-provision if traffic grows 20 req/s per minute while a new replica boots")
    ax.plot(ms, vals, color=BLUE, lw=2.2)
    for m in (1, 3, 8, 15, 25):
        v = vals[m]
        ax.plot([m], [v], "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
        ax.text(m - 0.4, v + 3.2, f"{v}", ha="right" if m > 1 else "center", fontsize=11, color=INK, fontweight="bold")
    ax.axhline(base, color=INK2, lw=1.4, ls="--")
    ax.text(13.5, base + 2.4, f"replicas needed at steady state today: {base}", fontsize=10.5, color=INK2)
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 92)
    ax.set_xlabel("cold start: schedule + pull + load weights + warm up (minutes)")
    ax.set_ylabel("headroom replicas")
    clean(ax, "y")
    save(fig, "fig5-coldstart-headroom.png")


def main():
    fig_architecture()
    fig_kv(kv_data())
    fig_routing(route_data(zipf=1.0), route_data(zipf=1.6))
    staleness = [0.05, 0.25, 0.5, 1.0, 2.0]
    series = {p: [route_data(policies=[p], stale=s)[p]["p50"] for s in staleness] for p in ("round-robin", "least-queue", "scored")}
    fig_staleness(staleness, series)
    fig_headroom()


if __name__ == "__main__":
    main()
