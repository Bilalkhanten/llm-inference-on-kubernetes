import statistics

import pytest

import captool
import kvcalc
import make_cover
import make_figures
import render
import routesim
import validate


# ---- kvcalc -----------------------------------------------------------------
def test_kv_bytes_per_token_llama8b_is_128kib():
    assert kvcalc.kv_bytes_per_token(32, 8, 128, 2) == 128 * 1024


def test_70b_does_not_fit_one_80gib_gpu_but_fits_tp2_barely():
    assert kvcalc.plan("Llama-3.1-70B", 1, 80, 0.90, 2, 2, 3.0, 8192) is None
    tp2 = kvcalc.plan("Llama-3.1-70B", 2, 80, 0.90, 2, 2, 3.0, 8192)
    assert 20_000 < tp2["tokens"] < 22_000          # ~3 GiB of KV budget
    assert tp2["conc"] < 3                           # ~2.6 concurrent 8k sequences


def test_fp8_kv_exactly_doubles_token_capacity():
    bf16 = kvcalc.plan("Llama-3.1-8B", 1, 80, 0.90, 2, 2, 3.0, 8192)
    fp8 = kvcalc.plan("Llama-3.1-8B", 1, 80, 0.90, 2, 1, 3.0, 8192)
    assert fp8["tokens"] / bf16["tokens"] == pytest.approx(2.0)


# ---- captool ----------------------------------------------------------------
def test_littles_law_defaults():
    n, usable, base = captool.steady_replicas(60, 6.0, 54, 0.70)
    assert n == 360 and usable == pytest.approx(37.8) and base == 10


def test_headroom_grows_roughly_linearly_with_cold_start():
    _, usable, _ = captool.steady_replicas(60, 6.0, 54, 0.70)
    heads = [captool.headroom_replicas(20, m, 6.0, usable)[2] for m in (1, 8, 15, 25)]
    assert heads == [4, 26, 48, 80]
    assert heads == sorted(heads)


# ---- routesim ---------------------------------------------------------------
def run(policy, **kw):
    kw.setdefault("lam", 80.0)
    kw.setdefault("n_req", 6000)
    return routesim.Sim(policy, 0, **kw).go()


def test_simulation_is_deterministic():
    a, b = run("scored"), run("scored")
    assert a["ttft"] == b["ttft"] and a["hit"] == b["hit"]


def test_cache_aware_policies_raise_prefix_hit_rate():
    rr, scored = run("round-robin"), run("scored")
    assert scored["hit"] > rr["hit"] + 0.2


def test_hard_prefix_affinity_melts_down_under_hot_prefix():
    kw = dict(zipf=1.6)
    hashed, lq = run("prefix-hash", **kw), run("least-queue", **kw)
    assert routesim.pct(hashed["ttft"], 99) > 10 * routesim.pct(lq["ttft"], 99)


def test_stale_metrics_hurt_least_queue_more_than_fresh_metrics():
    fresh = run("least-queue", stale=0.25)
    stale = run("least-queue", stale=2.0)
    assert statistics.median(stale["ttft"]) > statistics.median(fresh["ttft"])


# ---- manifests --------------------------------------------------------------
def test_builtin_manifests_validate_on_1_34():
    results = validate.validate_all("1.34")
    assert not [r for r in results if r[2] == "FAIL"]
    ok = [r for r in results if r[2] == "OK"]
    kinds = {r[1] for r in ok}
    assert {"ResourceClaimTemplate", "Deployment", "Namespace", "ClusterRole"} <= kinds


def test_crds_are_skipped_not_silently_passed():
    skipped = {r[1] for r in validate.validate_all("1.34") if r[2] == "SKIP"}
    assert {"InferencePool", "HTTPRoute", "ScaledObject"} <= skipped


# ---- renderer ---------------------------------------------------------------
def test_render_writes_a_png(tmp_path):
    out = tmp_path / "t.png"
    try:
        render.render(["$ echo hi", "OK  fine", "SKIP skipped"], "title", str(out))
    except SystemExit:
        pytest.skip("no monospace font available")
    assert out.stat().st_size > 1000


# ---- figures ----------------------------------------------------------------
def test_figure_data_comes_from_the_tools():
    rows = make_figures.kv_data()
    by = {(n, tp): (bf, f8) for n, tp, bf, f8 in rows}
    assert by[("Llama-3.1-70B", 1)] == (None, None)
    assert by[("Llama-3.1-70B", 2)][0] == pytest.approx(2.6, abs=0.05)
    assert by[("Llama-3.1-8B", 1)][1] == pytest.approx(2 * by[("Llama-3.1-8B", 1)][0])


# ---- cover ------------------------------------------------------------------
def test_cover_squares_add_up_to_the_whole_gpu_and_match_kvcalc():
    weights, overhead, kv, reserved, conc = make_cover.numbers()
    assert weights + overhead + kv + reserved == 80
    assert kv == 3 and conc == pytest.approx(2.6, abs=0.05)
