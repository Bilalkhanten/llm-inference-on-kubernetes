#!/usr/bin/env python3
"""Little's-law capacity + cold-start headroom.

Every input is an assumption. Replace them with your own measurements.
"""
import argparse
import math

COLD_STARTS_MIN = (1, 3, 8, 15, 25)


def steady_replicas(qps, e2e, seqs_per_replica, target_util):
    """Replicas needed in steady state. Little's law: in-flight N = lambda * W."""
    in_flight = qps * e2e
    usable = seqs_per_replica * target_util
    return in_flight, usable, math.ceil(in_flight / usable)


def headroom_replicas(ramp_qps_per_min, cold_start_min, e2e, usable_seqs):
    """Replicas to pre-provision so traffic that arrives *while a new replica boots* is absorbed."""
    added_qps = ramp_qps_per_min * cold_start_min
    extra = added_qps * e2e
    return added_qps, extra, math.ceil(extra / usable_seqs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qps", type=float, default=60, help="steady-state request rate")
    ap.add_argument("--e2e", type=float, default=6.0, help="mean end-to-end seconds per request at target load")
    ap.add_argument("--seqs-per-replica", type=float, default=54,
                    help="KV-bound concurrent seqs (from kvcalc.py, Llama-3.1-8B bf16 @8k)")
    ap.add_argument("--target-util", type=float, default=0.70, help="KV/slot occupancy you are willing to run at")
    ap.add_argument("--ramp-qps-per-min", type=float, default=20, help="how fast traffic can grow")
    a = ap.parse_args(argv)

    n, usable, base = steady_replicas(a.qps, a.e2e, a.seqs_per_replica, a.target_util)
    print(f"Little's law: in-flight seqs N = {a.qps:.0f} req/s x {a.e2e:.1f}s = {n:.0f}")
    print(f"usable seqs/replica = {a.seqs_per_replica:.0f} x {a.target_util:.0%} = {usable:.1f}"
          f"  ->  steady-state replicas = {base}\n")
    print("cold-start (schedule + pull + weights + warmup) vs. headroom you must pre-provision:")
    print(f"{'cold start':>11} {'traffic added while waiting':>29} {'extra in-flight seqs':>21}"
          f" {'headroom replicas':>18} {'floor (steady+headroom)':>24}")
    for m in COLD_STARTS_MIN:
        added_qps, extra, head = headroom_replicas(a.ramp_qps_per_min, m, a.e2e, usable)
        print(f"{m:>8} min {added_qps:>22.0f} req/s {extra:>21.0f} {head:>18d} {base + head:>24d}")


if __name__ == "__main__":
    main()
