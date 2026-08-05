"""Async stress test for the RetainStack prediction API.

Covers liveness, readiness, single-session prediction (dynamic batcher),
batch prediction at various sizes, a sustained throughput burst, and
Pydantic input-validation rejection checks.

Requirements (already in the project venv):
    httpx >= 0.28   (HTTP client)
    rich            (optional — pretty tables and spinners)

Usage:
    python stress_test.py [--host HOST] [--port PORT]
                          [--concurrency N] [--total N]
                          [--batch-sizes N [N ...]]
                          [--burst-duration S]
                          [--timeout S] [--seed N]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    _console = Console()
    RICH = True
except ImportError:
    RICH = False
    _console = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: uv add httpx  (or pip install httpx)")


# Valid months from the UCI Online Shoppers dataset
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ANSI codes used in the plain-text fallback
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _rand_session() -> dict[str, Any]:
    """Return one random valid SessionFeatures payload."""
    return {
        "Administrative":           random.randint(0, 20),
        "Administrative_Duration":  round(random.uniform(0.0, 600.0),  2),
        "Informational_Duration":   round(random.uniform(0.0, 300.0),  2),
        "ProductRelated":           random.randint(0, 200),
        "ProductRelated_Duration":  round(random.uniform(0.0, 5000.0), 2),
        "BounceRates":              round(random.uniform(0.0, 1.0),    4),
        "ExitRates":                round(random.uniform(0.0, 1.0),    4),
        "PageValues":               round(random.uniform(0.0, 400.0),  2),
        "Month":                    random.choice(_MONTHS),
    }


@dataclass
class EndpointStats:
    """Aggregated latency and error counts for one test phase."""

    name: str
    total_requests: int = 0
    succeeded: int = 0
    failed: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.succeeded / self.total_requests * 100

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0


# Core request helpers

async def _get(client: httpx.AsyncClient, url: str, stats: EndpointStats, *, timeout: float) -> None:
    t0 = time.perf_counter()
    try:
        resp = await client.get(url, timeout=timeout)
        elapsed = (time.perf_counter() - t0) * 1000
        stats.latencies_ms.append(elapsed)
        stats.total_requests += 1
        if resp.is_success:
            stats.succeeded += 1
        else:
            stats.failed += 1
            stats.errors.append(f"HTTP {resp.status_code}")
    except Exception as exc:
        stats.latencies_ms.append((time.perf_counter() - t0) * 1000)
        stats.total_requests += 1
        stats.failed += 1
        stats.errors.append(str(exc)[:80])


async def _post(client: httpx.AsyncClient, url: str, payload: dict,
                stats: EndpointStats, *, timeout: float) -> None:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        elapsed = (time.perf_counter() - t0) * 1000
        stats.latencies_ms.append(elapsed)
        stats.total_requests += 1
        if resp.is_success:
            stats.succeeded += 1
        else:
            stats.failed += 1
            stats.errors.append(f"HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        stats.latencies_ms.append((time.perf_counter() - t0) * 1000)
        stats.total_requests += 1
        stats.failed += 1
        stats.errors.append(str(exc)[:80])


# Semaphore-limited worker pools

async def _concurrent_get(base_url: str, path: str, total: int,
                           concurrency: int, timeout: float) -> EndpointStats:
    stats = EndpointStats(name=f"GET {path}")
    sem = asyncio.Semaphore(concurrency)
    url = f"{base_url}{path}"

    async def _worker() -> None:
        async with sem:
            async with httpx.AsyncClient() as client:
                await _get(client, url, stats, timeout=timeout)

    await asyncio.gather(*[asyncio.create_task(_worker()) for _ in range(total)])
    return stats


async def _concurrent_post(base_url: str, path: str, payload_fn,
                            total: int, concurrency: int, timeout: float) -> EndpointStats:
    stats = EndpointStats(name=f"POST {path}")
    sem = asyncio.Semaphore(concurrency)
    url = f"{base_url}{path}"

    async def _worker() -> None:
        async with sem:
            async with httpx.AsyncClient() as client:
                await _post(client, url, payload_fn(), stats, timeout=timeout)

    await asyncio.gather(*[asyncio.create_task(_worker()) for _ in range(total)])
    return stats


async def _throughput_burst(base_url: str, path: str, payload_fn,
                             concurrency: int, duration_s: float,
                             timeout: float) -> tuple[EndpointStats, float]:
    """Fire requests as fast as possible for duration_s seconds."""
    stats = EndpointStats(name=f"POST {path}")
    sem = asyncio.Semaphore(concurrency)
    url = f"{base_url}{path}"
    deadline = time.perf_counter() + duration_s
    tasks: list[asyncio.Task] = []

    async def _worker() -> None:
        async with sem:
            async with httpx.AsyncClient() as client:
                await _post(client, url, payload_fn(), stats, timeout=timeout)

    t_start = time.perf_counter()
    while time.perf_counter() < deadline:
        tasks.append(asyncio.create_task(_worker()))
        await asyncio.sleep(0)

    await asyncio.gather(*tasks)
    return stats, time.perf_counter() - t_start


# Helpers for display

async def _run_phase(label: str, coro) -> Any:
    """Run a coroutine, optionally wrapping it with a Rich progress spinner."""
    if RICH:
        with Progress(SpinnerColumn(),
                      TextColumn(f"[bold cyan]{label}[/bold cyan]"),
                      BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
                      console=_console, transient=True) as progress:
            tid = progress.add_task(label, total=None)
            result = await coro
            progress.update(tid, completed=1, total=1)
        return result
    print(f"  -> {label} ...", end="", flush=True)
    result = await coro
    print(" done")
    return result


def _section(text: str) -> None:
    if RICH:
        _console.rule(f"[bold yellow]{text}[/bold yellow]")
    else:
        print(f"\n{_YELLOW}-- {text} --{_RESET}")


def _print_table(all_stats: list[EndpointStats]) -> None:
    if RICH:
        table = Table(title="Stress Test Results", show_lines=True, header_style="bold magenta")
        for col, kw in [
            ("Endpoint",  dict(style="cyan", no_wrap=True)),
            ("Total",     dict(justify="right")),
            ("OK",        dict(justify="right", style="green")),
            ("Fail",      dict(justify="right", style="red")),
            ("Success%",  dict(justify="right")),
            ("Avg(ms)",   dict(justify="right")),
            ("P50(ms)",   dict(justify="right")),
            ("P95(ms)",   dict(justify="right")),
            ("P99(ms)",   dict(justify="right")),
            ("Min(ms)",   dict(justify="right")),
            ("Max(ms)",   dict(justify="right")),
        ]:
            table.add_column(col, **kw)

        for s in all_stats:
            pct = s.success_rate
            pct_str = (f"[green]{pct:.1f}%[/green]" if pct >= 95
                       else f"[red]{pct:.1f}%[/red]")
            table.add_row(s.name, str(s.total_requests), str(s.succeeded),
                          str(s.failed), pct_str,
                          f"{s.avg_ms:.1f}", f"{s.p50_ms:.1f}",
                          f"{s.p95_ms:.1f}", f"{s.p99_ms:.1f}",
                          f"{s.min_ms:.1f}", f"{s.max_ms:.1f}")
        _console.print(table)
    else:
        hdr = (f"{'Endpoint':<44} {'Total':>6} {'OK':>6} {'Fail':>6} "
               f"{'Succ%':>6} {'Avg':>7} {'P50':>7} {'P95':>7} {'P99':>7} "
               f"{'Min':>7} {'Max':>7}")
        print(f"\n{_BOLD}{hdr}{_RESET}")
        print("-" * len(hdr))
        for s in all_stats:
            c = _GREEN if s.success_rate >= 95 else _RED
            print(f"{s.name:<44} {s.total_requests:>6} {s.succeeded:>6} {s.failed:>6} "
                  f"{c}{s.success_rate:>5.1f}%{_RESET} "
                  f"{s.avg_ms:>7.1f} {s.p50_ms:>7.1f} {s.p95_ms:>7.1f} "
                  f"{s.p99_ms:>7.1f} {s.min_ms:>7.1f} {s.max_ms:>7.1f}")


def _print_errors(stats: EndpointStats, max_show: int = 5) -> None:
    if not stats.errors:
        return
    counts: dict[str, int] = {}
    for e in stats.errors:
        counts[e] = counts.get(e, 0) + 1
    label = f"  Top errors for {stats.name}"
    if RICH:
        _console.print(f"[bold red]{label}[/bold red]")
        for msg, n in list(counts.items())[:max_show]:
            _console.print(f"    [{n}x] {msg}", style="red")
    else:
        print(f"{_RED}{label}{_RESET}")
        for msg, n in list(counts.items())[:max_show]:
            print(f"  [{n}x] {msg}")


# Main orchestration
async def run_stress_test(args: argparse.Namespace) -> int:
    base_url = f"http://{args.host}:{args.port}"

    header = (f"RetainStack Stress Test  |  {base_url}\n"
              f"Concurrency: {args.concurrency}  |  "
              f"Requests/endpoint: {args.total}  |  Timeout: {args.timeout}s")
    if RICH:
        _console.print(Panel(f"[bold white]{header}[/bold white]", style="blue"))
    else:
        print(f"\n{_BOLD}{_CYAN}{'=' * 60}\n  {header}\n{'=' * 60}{_RESET}")

    all_stats: list[EndpointStats] = []

    # Phase 1 — liveness probe
    _section("Phase 1 -- Liveness Probe  GET /api/v1/health")
    all_stats.append(await _run_phase(
        "Health checks",
        _concurrent_get(base_url, "/api/v1/health",
                        args.total, args.concurrency, args.timeout),
    ))

    # Phase 2 — readiness probe
    _section("Phase 2 -- Readiness Probe  GET /api/v1/ready")
    all_stats.append(await _run_phase(
        "Readiness checks",
        _concurrent_get(base_url, "/api/v1/ready",
                        args.total, args.concurrency, args.timeout),
    ))

    # Phase 3 — single-session prediction (exercises the dynamic batcher)
    _section("Phase 3 -- Single Predict  POST /api/v1/predict  (Dynamic Batcher)")
    all_stats.append(await _run_phase(
        f"Single predict ({args.total} requests, concurrency={args.concurrency})",
        _concurrent_post(base_url, "/api/v1/predict", _rand_session,
                         args.total, args.concurrency, args.timeout),
    ))

    # Phase 4 — batch predict at various batch sizes
    _section("Phase 4 -- Batch Predict  POST /api/v1/predict/batch")
    for bs in args.batch_sizes:
        def _make_batch(n: int = bs) -> dict:
            return {"sessions": [_rand_session() for _ in range(n)]}

        s = await _run_phase(
            f"Batch size {bs:>3}  ({args.total} requests)",
            _concurrent_post(base_url, "/api/v1/predict/batch", _make_batch,
                             args.total, args.concurrency, args.timeout),
        )
        s.name = f"POST /api/v1/predict/batch [bs={bs}]"
        all_stats.append(s)

    # Phase 5 — max-load batch (500 sessions — the API hard cap)
    _section("Phase 5 -- Max-Load Batch  POST /api/v1/predict/batch [bs=500]")
    max_total = max(1, args.total // 5)

    def _max_batch() -> dict:
        return {"sessions": [_rand_session() for _ in range(500)]}

    s = await _run_phase(
        f"Max-load batch  ({max_total} requests x 500 sessions)",
        _concurrent_post(base_url, "/api/v1/predict/batch", _max_batch,
                         max_total, min(args.concurrency, 20), args.timeout * 3),
    )
    s.name = "POST /api/v1/predict/batch [bs=500 MAX]"
    all_stats.append(s)

    # Phase 6 — sustained throughput burst
    _section(f"Phase 6 -- Throughput Burst  POST /api/v1/predict  ({args.burst_duration}s)")
    burst_stats, elapsed = await _run_phase(
        f"Throughput burst ({args.burst_duration}s wall-clock)",
        _throughput_burst(base_url, "/api/v1/predict", _rand_session,
                          args.concurrency, args.burst_duration, args.timeout),
    )
    rps = burst_stats.total_requests / elapsed if elapsed > 0 else 0.0
    burst_stats.name = f"POST /api/v1/predict [burst {args.burst_duration}s]"
    all_stats.append(burst_stats)

    # Phase 7 — Pydantic input-validation rejection
    _section("Phase 7 -- Invalid Payload Rejection  POST /api/v1/predict")
    invalid_payloads = [
        {},                                             # all fields missing
        {"Month": "Nov"},                               # only month field
        {**_rand_session(), "BounceRates": 999.9},     # out-of-range float
        {**_rand_session(), "Month": "BADMONTH"},       # unknown month string
        {**_rand_session(), "ProductRelated": -1},      # negative integer
    ]
    invalid_stats = EndpointStats(name="POST /api/v1/predict [invalid payloads]")
    async with httpx.AsyncClient(base_url=base_url) as client:
        for payload in invalid_payloads:
            t0 = time.perf_counter()
            try:
                resp = await client.post("/api/v1/predict", json=payload,
                                         timeout=args.timeout)
                invalid_stats.latencies_ms.append((time.perf_counter() - t0) * 1000)
                invalid_stats.total_requests += 1
                # HTTP 422 Unprocessable Entity is the expected rejection status
                if resp.status_code == 422:
                    invalid_stats.succeeded += 1
                else:
                    invalid_stats.failed += 1
                    invalid_stats.errors.append(
                        f"Expected 422 for {list(payload.keys())}, got {resp.status_code}"
                    )
            except Exception as exc:
                invalid_stats.total_requests += 1
                invalid_stats.failed += 1
                invalid_stats.errors.append(str(exc)[:80])
    invalid_stats.name += f"  ({invalid_stats.total_requests} probes, expect HTTP 422)"
    all_stats.append(invalid_stats)

    # Print results table
    _section("Results Summary")
    _print_table(all_stats)

    # Throughput summary line
    burst_line = (f"Throughput burst:  {rps:.1f} req/s over {elapsed:.2f}s  "
                  f"({burst_stats.succeeded} OK / {burst_stats.failed} failed)")
    if RICH:
        _console.print(f"\n[bold cyan]{burst_line}[/bold cyan]")
    else:
        print(f"\n{_CYAN}{burst_line}{_RESET}")

    # Error details per phase
    for s in all_stats:
        _print_errors(s)

    # Overall pass/fail — invalid-payload phase excluded from the gate
    passed = all(s.success_rate >= 95 for s in all_stats
                 if "invalid" not in s.name.lower())
    if passed:
        msg = "All service checks PASSED (>= 95% success rate)"
        if RICH:
            _console.print(Panel(f"[bold green]{msg}[/bold green]", style="green"))
        else:
            print(f"\n{_GREEN}{_BOLD}{msg}{_RESET}\n")
        return 0

    msg = "One or more service checks FAILED (< 95% success rate)"
    if RICH:
        _console.print(Panel(f"[bold red]{msg}[/bold red]", style="red"))
    else:
        print(f"\n{_RED}{_BOLD}{msg}{_RESET}\n")
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stress_test.py",
        description="Stress-test the RetainStack prediction API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",           default="127.0.0.1")
    parser.add_argument("--port",           type=int,   default=8000)
    parser.add_argument("--concurrency",    type=int,   default=50,
                        help="Max concurrent in-flight requests")
    parser.add_argument("--total",          type=int,   default=200,
                        help="Requests per endpoint phase")
    parser.add_argument("--batch-sizes",    type=int,   nargs="+",
                        default=[1, 10, 50, 100], metavar="N",
                        help="Batch sizes tested in Phase 4")
    parser.add_argument("--burst-duration", type=float, default=10.0,
                        help="Wall-clock seconds for the throughput burst")
    parser.add_argument("--timeout",        type=float, default=30.0,
                        help="Per-request timeout in seconds")
    parser.add_argument("--seed",           type=int,   default=None,
                        help="Random seed for reproducible payloads")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    sys.exit(asyncio.run(run_stress_test(args)))
