#!/usr/bin/env python3
"""
Toy discrete-event simulation of LLM request routing across N replicas.
NOT a benchmark of vLLM or of any real endpoint picker: it models just enough
(slots, FIFO queue, per-replica LRU prefix cache, stale metrics) to show *why*
routing policy matters. Every constant below is an assumption.
"""
import heapq, random, math, argparse, statistics as st
from collections import OrderedDict

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs)-1, int(p/100*len(xs)))]

class Sim:
    def __init__(s, policy, seed, R=8, slots=32, n_prefix=200, cache_cap=40, zipf=1.0, share=0.8,
                 prefix_len=3000, lam=40.0, n_req=20000, stale=0.25,
                 t_pre=0.08e-3, t_dec0=8e-3, t_dec_k=0.12e-3, w_q=1.0, w_p=1.5):
        s.p, s.R, s.slots, s.stale = policy, R, slots, stale
        s.rng = random.Random(seed)
        s.cache = [OrderedDict() for _ in range(R)]; s.cap = cache_cap
        s.run = [0]*R; s.wait = [[] for _ in range(R)]; s.snap = [0]*R; s.snap_t = -1
        s.t_pre, s.t_dec0, s.t_dec_k, s.w_q, s.w_p = t_pre, t_dec0, t_dec_k, w_q, w_p
        w = [1/(i+1)**zipf for i in range(n_prefix)]; tot = sum(w)
        s.cdf = []; a = 0
        for x in w: a += x/tot; s.cdf.append(a)
        s.share, s.plen, s.lam, s.n, s.rr = share, prefix_len, lam, n_req, 0
        s.ttft, s.e2e, s.hit_tok, s.tot_tok, s.busy = [], [], 0, 0, 0.0

    def pick_prefix(s):
        if s.rng.random() > s.share: return None
        u = s.rng.random()
        for i, c in enumerate(s.cdf):
            if u <= c: return i
        return len(s.cdf)-1

    def choose(s, now, pid):
        load = [s.run[i] + len(s.wait[i]) for i in range(s.R)]
        if now - s.snap_t >= s.stale: s.snap, s.snap_t = load[:], now   # stale metrics, like a scraped /metrics
        if s.p == "round-robin": s.rr += 1; return s.rr % s.R
        if s.p == "random": return s.rng.randrange(s.R)
        if s.p == "least-queue": return min(range(s.R), key=lambda i: s.snap[i])
        if s.p == "prefix-hash": return (pid if pid is not None else s.rng.randrange(10**6)) % s.R
        if s.p == "scored":   # EPP-style: queue pressure vs. prefix-cache locality
            def score(i):
                hit = pid is not None and pid in s.cache[i]
                return s.w_q * s.snap[i] / s.slots - (s.w_p if hit else 0.0)
            return min(range(s.R), key=score)

    def start(s, now, i, req, ev):
        arr, pid, suf, out = req
        pre = suf
        if pid is not None:
            if pid in s.cache[i]: s.cache[i].move_to_end(pid); s.hit_tok += s.plen
            else:
                pre += s.plen; s.cache[i][pid] = 1
                if len(s.cache[i]) > s.cap: s.cache[i].popitem(last=False)
            s.tot_tok += s.plen
        t_pre = pre * s.t_pre
        s.run[i] += 1
        t_dec = out * (s.t_dec0 + s.t_dec_k * s.run[i])
        s.ttft.append(now + t_pre - arr); s.e2e.append(now + t_pre + t_dec - arr); s.busy += t_pre + t_dec
        heapq.heappush(ev, (now + t_pre + t_dec, 1, i, None))

    def go(s):
        ev, t = [], 0.0
        for _ in range(s.n):
            t += s.rng.expovariate(s.lam)
            suf = max(20, int(s.rng.lognormvariate(math.log(250), 0.6)))
            out = max(10, int(s.rng.lognormvariate(math.log(200), 0.7)))
            heapq.heappush(ev, (t, 0, None, (t, s.pick_prefix(), suf, out)))
        end = 0
        while ev:
            now, kind, i, req = heapq.heappop(ev); end = now
            if kind == 0:
                i = s.choose(now, req[1])
                if s.run[i] < s.slots: s.start(now, i, req, ev)
                else: s.wait[i].append(req)
            else:
                s.run[i] -= 1
                if s.wait[i]: s.start(now, i, s.wait[i].pop(0), ev)
        return dict(ttft=s.ttft, e2e=s.e2e, hit=s.hit_tok/max(1, s.tot_tok), util=s.busy/(s.R*s.slots*end))

def run(label, **kw):
    print(f"\n== {label}")
    print(f"{'policy':<14}{'TTFT p50':>10}{'p95':>9}{'p99':>9}{'E2E p99':>10}{'prefix hit':>12}{'slot util':>11}")
    for pol in ["random", "round-robin", "least-queue", "prefix-hash", "scored"]:
        res = [Sim(pol, seed, **kw).go() for seed in range(5)]
        f = lambda k, p: st.mean(pct(r[k], p) for r in res)
        print(f"{pol:<14}{f('ttft',50):>9.2f}s{f('ttft',95):>8.2f}s{f('ttft',99):>8.2f}s{f('e2e',99):>9.2f}s"
              f"{st.mean(r['hit'] for r in res):>11.0%}{st.mean(r['util'] for r in res):>11.0%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lam", type=float, default=40.0)
    ap.add_argument("--stale", type=float, default=0.25, help="seconds between metric snapshots seen by the router")
    a = ap.parse_args()
    print("8 replicas x 32 slots | 200 shared prefixes (3000 tok) | cache: 40 prefixes/replica")
    print(f"metrics staleness {int(a.stale*1000)}ms | mean of 5 seeds x 20k requests each")
    run(f"A) mild skew (Zipf s=1.0), {a.lam:.0f} req/s", lam=a.lam, zipf=1.0, stale=a.stale)
    run(f"B) hot-prefix skew (Zipf s=1.6), {a.lam:.0f} req/s", lam=a.lam, zipf=1.6, stale=a.stale)
