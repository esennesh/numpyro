# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Prototype fixed-window relaxed counts around a quantile anchor.

This module deliberately lives outside ``numpyro``: it compares an experimental
direct-CDF formulation with the production adaptive accumulator without changing
the public implementation. Run it from the repository root with::

    python -m benchmarks.diag_sgd_anchor_prototype

The prototype covers every unbounded count family handled by
``numpyro.contrib.diag_sgd``: Poisson, Geometric, GammaPoisson (including the
NegativeBinomial parameterizations), and zero-inflated wrappers around those
families. It enables float64 by default because count CDFs can lose enough
accuracy in float32 to select the wrong anchor in their tails. Pass ``--x32``
to inspect that precision loss.

Finite-support DSGD families are intentionally absent: their production path
already evaluates a static, exact-size CDF grid and has no adaptive-horizon
straggler for an anchor to remove.

Related implementation and report:

* :func:`numpyro.contrib.diag_sgd.adaptive_relaxed_count` is the existing DSGD
  relaxed-count implementation against which this prototype is benchmarked.
* `esennesh/numpyro issue #1
  <https://github.com/esennesh/numpyro/issues/1>`_ reports the heterogeneous-rate
  ``vmap`` straggler that motivated the anchored formulation.

The binary-search anchor uses a fixed number of CDF evaluations and is detached
from automatic differentiation. The Cornish--Fisher anchor is included to test
whether its cheaper approximation is accurate enough to select the same local
correction window.
"""

import argparse
import csv
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import poisson as scipy_poisson

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, betaln, gammaincc, gammaln, ndtri

from numpyro.contrib.diag_sgd import adaptive_relaxed_count
import numpyro.distributions as dist

__all__ = [
    "poisson_anchor_binary_search",
    "poisson_anchor_cornish_fisher",
    "count_anchor_binary_search",
    "count_anchor_cornish_fisher",
    "anchored_relaxed_count_direct",
    "anchored_relaxed_count_recurrence",
    "poisson_anchored_relaxed_count_direct",
    "poisson_anchored_relaxed_count_recurrence",
]

DEFAULT_ETA = 0.01
DEFAULT_MAX_COUNT = 100_000
DEFAULT_REFERENCE_WIDTH = 1_024
DEFAULT_REPEATS = 3
DEFAULT_SAMPLES = 257
DEFAULT_WIDTH = 256


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help="write one CSV per benchmark section to this directory",
    )
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--max-count", type=int, default=DEFAULT_MAX_COUNT)
    parser.add_argument(
        "--rates", type=float, nargs="+", default=[1e-3, 1.0, 10.0, 100.0, 1e4]
    )
    parser.add_argument(
        "--diagnostic-lanes",
        type=int,
        default=33,
        help="number of timing lanes, including the straggler, used for accuracy diagnostics",
    )
    parser.add_argument("--reference-width", type=int, default=DEFAULT_REFERENCE_WIDTH)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--timing-lanes", type=int, default=4_096)
    parser.add_argument("--timing-small-mean", type=float, default=1e-3)
    parser.add_argument("--timing-large-mean", type=float, default=1e4)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--x32", action="store_true")
    return parser


def _gradient(fn, rate):
    return jax.grad(lambda value: jnp.sum(fn(value)))(rate)


def _max_abs(actual, expected):
    return float(jnp.max(jnp.abs(actual - expected)))


def _write_csv(directory, filename, header, rows):
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _time(fn, repeats):
    result = fn()
    jax.block_until_ready(result)
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    jax.block_until_ready(result)
    return (time.perf_counter() - start) / repeats


def _benchmark_other_families(args, dtype, u):
    """Exercise the generic prototype on every other supported count family."""
    concentration = jnp.asarray(2.5, dtype=dtype)
    gate = jnp.asarray(0.3, dtype=dtype)
    cases = [
        (
            "geometric",
            (0.2,),
            lambda value: dist.GeometricProbs(value),
        ),
        (
            "gamma-poisson",
            (1.0,),
            lambda value: dist.GammaPoisson(concentration, value),
        ),
        (
            "zero-inflated-poisson",
            (4.0,),
            lambda value: dist.ZeroInflatedPoisson(gate, value),
        ),
        (
            "zero-inflated-geometric",
            (0.2,),
            lambda value: dist.ZeroInflatedDistribution(
                dist.GeometricProbs(value), gate=gate
            ),
        ),
        (
            "zero-inflated-gamma-poisson",
            (1.0,),
            lambda value: dist.ZeroInflatedDistribution(
                dist.GammaPoisson(concentration, value), gate=gate
            ),
        ),
    ]

    print("\nOther supported unbounded families")
    rows = []
    print(
        "family\tparameter\tCF anchor\tcurrent value\trecurrence value\t"
        "CF value\tcurrent grad\trecurrence grad\tCF grad"
    )
    for family, parameter_values, make_dist in cases:
        for parameter_value in parameter_values:
            parameter = jnp.full_like(u, parameter_value)

            def recurrence_fn(value):
                base_dist = make_dist(value)
                anchor = count_anchor_binary_search(base_dist, u, args.max_count)
                return anchored_relaxed_count_recurrence(
                    base_dist, anchor, args.eta, u, args.width
                )

            def cornish_fisher_fn(value):
                base_dist = make_dist(value)
                anchor = count_anchor_cornish_fisher(base_dist, u)
                return anchored_relaxed_count_recurrence(
                    base_dist, anchor, args.eta, u, args.width
                )

            def current_fn(value):
                return adaptive_relaxed_count(make_dist(value), u, args.eta)

            def reference_fn(value):
                base_dist = make_dist(value)
                anchor = count_anchor_binary_search(base_dist, u, args.max_count)
                return anchored_relaxed_count_direct(
                    base_dist, anchor, args.eta, u, args.reference_width
                )

            base_dist = make_dist(parameter)
            anchor = count_anchor_binary_search(base_dist, u, args.max_count)
            cf_anchor = count_anchor_cornish_fisher(base_dist, u)
            reference = reference_fn(parameter)
            reference_gradient = _gradient(reference_fn, parameter)
            row = (
                family,
                parameter_value,
                _max_abs(cf_anchor, anchor),
                _max_abs(current_fn(parameter), reference),
                _max_abs(recurrence_fn(parameter), reference),
                _max_abs(cornish_fisher_fn(parameter), reference),
                _max_abs(_gradient(current_fn, parameter), reference_gradient),
                _max_abs(_gradient(recurrence_fn, parameter), reference_gradient),
                _max_abs(_gradient(cornish_fisher_fn, parameter), reference_gradient),
            )
            rows.append(row)
            print(
                f"{row[0]}\t{row[1]:g}\t{row[2]:.0f}\t{row[3]:.3g}\t"
                f"{row[4]:.3g}\t{row[5]:.3g}\t{row[6]:.3g}\t"
                f"{row[7]:.3g}\t{row[8]:.3g}"
            )

    _write_csv(
        args.csv_dir,
        "other_families_accuracy.csv",
        (
            "family",
            "parameter",
            "cf_anchor_max_abs_error",
            "current_value_max_abs_error",
            "recurrence_value_max_abs_error",
            "cf_value_max_abs_error",
            "current_gradient_max_abs_error",
            "recurrence_gradient_max_abs_error",
            "cf_gradient_max_abs_error",
        ),
        rows,
    )


def _benchmark_timings(args, dtype):
    """Time heterogeneous batches for each non-aliased unbounded family."""
    concentration = jnp.asarray(2.5, dtype=dtype)
    gate = jnp.asarray(0.3, dtype=dtype)
    small_mean = args.timing_small_mean
    large_mean = args.timing_large_mean
    cases = [
        ("poisson", small_mean, large_mean, lambda value: dist.Poisson(value)),
        (
            "geometric",
            1.0 / (small_mean + 1.0),
            1.0 / (large_mean + 1.0),
            lambda value: dist.GeometricProbs(value),
        ),
        (
            "gamma-poisson",
            float(concentration / small_mean),
            float(concentration / large_mean),
            lambda value: dist.GammaPoisson(concentration, value),
        ),
        (
            "zero-inflated-poisson",
            small_mean,
            large_mean,
            lambda value: dist.ZeroInflatedPoisson(gate, value),
        ),
        (
            "zero-inflated-geometric",
            1.0 / (small_mean + 1.0),
            1.0 / (large_mean + 1.0),
            lambda value: dist.ZeroInflatedDistribution(
                dist.GeometricProbs(value), gate=gate
            ),
        ),
        (
            "zero-inflated-gamma-poisson",
            float(concentration / small_mean),
            float(concentration / large_mean),
            lambda value: dist.ZeroInflatedDistribution(
                dist.GammaPoisson(concentration, value), gate=gate
            ),
        ),
    ]

    diagnostic_lanes = min(args.diagnostic_lanes, args.timing_lanes)
    print("\nMixed-parameter timing")
    print(
        "family\ttypical parameter\tstraggler parameter\tmethod\tforward ms\t"
        "forward+backward ms\tvalue error\tgradient error\treference gradient\t"
        "relative gradient error"
    )
    print(
        f"Errors use the first {diagnostic_lanes} lanes and direct CDF width "
        f"{args.reference_width}."
    )
    timing_u = jax.random.uniform(jax.random.key(0), (args.timing_lanes,), dtype=dtype)
    rows = []
    timing_header = (
        "family",
        "typical_parameter",
        "straggler_parameter",
        "method",
        "forward_ms",
        "forward_backward_ms",
        "value_max_abs_error_vs_direct_reference",
        "gradient_max_abs_error_vs_direct_reference",
        "reference_gradient_max_abs",
        "gradient_max_relative_error_vs_direct_reference",
    )

    for family, typical, straggler, make_dist in cases:
        parameter = (
            jnp.full(args.timing_lanes, typical, dtype=dtype).at[0].set(straggler)
        )

        def binary_timing(value):
            base_dist = make_dist(value)
            anchor = count_anchor_binary_search(base_dist, timing_u, args.max_count)
            return anchored_relaxed_count_recurrence(
                base_dist, anchor, args.eta, timing_u, args.width
            )

        def cornish_fisher_timing(value):
            base_dist = make_dist(value)
            anchor = count_anchor_cornish_fisher(base_dist, timing_u)
            return anchored_relaxed_count_recurrence(
                base_dist, anchor, args.eta, timing_u, args.width
            )

        def current_timing(value):
            return adaptive_relaxed_count(make_dist(value), timing_u, args.eta)

        diagnostic_u = timing_u[:diagnostic_lanes]
        diagnostic_parameter = parameter[:diagnostic_lanes]

        def direct_reference(value):
            base_dist = make_dist(value)
            anchor = count_anchor_binary_search(base_dist, diagnostic_u, args.max_count)
            return anchored_relaxed_count_direct(
                base_dist, anchor, args.eta, diagnostic_u, args.reference_width
            )

        reference_forward = jax.jit(direct_reference)
        reference_backward = jax.jit(
            jax.value_and_grad(lambda value: direct_reference(value).sum())
        )
        reference_value = reference_forward(diagnostic_parameter)
        _, reference_gradient = reference_backward(diagnostic_parameter)
        reference_gradient_magnitude = float(jnp.max(jnp.abs(reference_gradient)))

        for method, fn in [
            ("binary", binary_timing),
            ("cornish-fisher", cornish_fisher_timing),
            ("current", current_timing),
        ]:
            forward = jax.jit(fn)
            forward_backward = jax.jit(
                jax.value_and_grad(lambda value: fn(value).sum())
            )
            forward_ms = 1_000 * _time(lambda: forward(parameter), args.repeats)
            forward_backward_ms = 1_000 * _time(
                lambda: forward_backward(parameter), args.repeats
            )
            actual_value = forward(parameter)[:diagnostic_lanes]
            _, actual_gradient = forward_backward(parameter)
            actual_gradient = actual_gradient[:diagnostic_lanes]
            value_error = _max_abs(actual_value, reference_value)
            gradient_error = _max_abs(actual_gradient, reference_gradient)
            relative_gradient_error = gradient_error / max(
                reference_gradient_magnitude, float(jnp.finfo(dtype).tiny)
            )
            row = (
                family,
                typical,
                straggler,
                method,
                forward_ms,
                forward_backward_ms,
                value_error,
                gradient_error,
                reference_gradient_magnitude,
                relative_gradient_error,
            )
            rows.append(row)
            _write_csv(args.csv_dir, "timings.csv", timing_header, rows)
            print(
                f"{row[0]}\t{row[1]:.3g}\t{row[2]:.3g}\t{row[3]}\t{row[4]:.1f}\t"
                f"{row[5]:.1f}\t{row[6]:.3g}\t{row[7]:.3g}\t{row[8]:.3g}\t"
                f"{row[9]:.3g}"
            )
        jax.clear_caches()


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if not args.x32:
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float32 if args.x32 else jnp.float64
    u = jnp.linspace(0.001, 0.999, args.samples, dtype=dtype)

    print(f"Quantile-anchor accuracy ({dtype})")
    print("rate\tbinary exact\tbinary max |error|\tCF exact\tCF max |error|")
    anchor_rows = []
    for rate_value in args.rates:
        rate = jnp.full_like(u, rate_value)
        binary = poisson_anchor_binary_search(rate, u, args.max_count)
        cornish_fisher = poisson_anchor_cornish_fisher(rate, u)
        exact = scipy_poisson.ppf(np.asarray(u), rate_value).astype(int)
        binary_error = np.abs(np.asarray(binary) - exact)
        cornish_fisher_error = np.abs(np.asarray(cornish_fisher) - exact)
        row = (
            rate_value,
            float(np.mean(binary_error == 0)),
            int(np.max(binary_error)),
            float(np.mean(cornish_fisher_error == 0)),
            int(np.max(cornish_fisher_error)),
        )
        anchor_rows.append(row)
        print(
            f"{rate_value:g}\t{float(np.mean(binary_error == 0)):.3f}\t"
            f"{int(np.max(binary_error))}\t"
            f"{float(np.mean(cornish_fisher_error == 0)):.3f}\t"
            f"{int(np.max(cornish_fisher_error))}"
        )
    _write_csv(
        args.csv_dir,
        "poisson_anchor_accuracy.csv",
        (
            "rate",
            "binary_exact_fraction",
            "binary_max_abs_error",
            "cf_exact_fraction",
            "cf_max_abs_error",
        ),
        anchor_rows,
    )

    print("\nRelaxed-count accuracy")
    print(
        "rate\tcurrent value\tbinary value\tCF value\t"
        "current grad\tbinary grad\tCF grad"
    )
    relaxed_rows = []
    for rate_value in args.rates:
        rate = jnp.full_like(u, rate_value)

        def binary_fn(value):
            anchor = poisson_anchor_binary_search(value, u, args.max_count)
            return poisson_anchored_relaxed_count_recurrence(
                anchor, args.eta, value, u, args.width
            )

        def cornish_fisher_fn(value):
            anchor = poisson_anchor_cornish_fisher(value, u)
            return poisson_anchored_relaxed_count_recurrence(
                anchor, args.eta, value, u, args.width
            )

        def current_fn(value):
            return adaptive_relaxed_count(dist.Poisson(value), u, args.eta)

        def reference_fn(value):
            anchor = poisson_anchor_binary_search(value, u, args.max_count)
            return poisson_anchored_relaxed_count_direct(
                anchor, args.eta, value, u, args.reference_width
            )

        binary = binary_fn(rate)
        cornish_fisher = cornish_fisher_fn(rate)
        current = current_fn(rate)
        reference = reference_fn(rate)
        binary_gradient = _gradient(binary_fn, rate)
        cornish_fisher_gradient = _gradient(cornish_fisher_fn, rate)
        current_gradient = _gradient(current_fn, rate)
        reference_gradient = _gradient(reference_fn, rate)
        relaxed_rows.append(
            (
                rate_value,
                _max_abs(current, reference),
                _max_abs(binary, reference),
                _max_abs(cornish_fisher, reference),
                _max_abs(current_gradient, reference_gradient),
                _max_abs(binary_gradient, reference_gradient),
                _max_abs(cornish_fisher_gradient, reference_gradient),
            )
        )
        print(
            f"{rate_value:g}\t{_max_abs(current, reference):.3g}\t"
            f"{_max_abs(binary, reference):.3g}\t"
            f"{_max_abs(cornish_fisher, reference):.3g}\t"
            f"{_max_abs(current_gradient, reference_gradient):.3g}\t"
            f"{_max_abs(binary_gradient, reference_gradient):.3g}\t"
            f"{_max_abs(cornish_fisher_gradient, reference_gradient):.3g}"
        )
    _write_csv(
        args.csv_dir,
        "poisson_relaxed_count_accuracy.csv",
        (
            "rate",
            "current_value_max_abs_error",
            "binary_value_max_abs_error",
            "cf_value_max_abs_error",
            "current_gradient_max_abs_error",
            "binary_gradient_max_abs_error",
            "cf_gradient_max_abs_error",
        ),
        relaxed_rows,
    )

    _benchmark_other_families(args, dtype, u)

    _benchmark_timings(args, dtype)


@jax.custom_jvp
def _betainc_in_concentration(concentration, count, probability):
    """``betainc`` with the concentration derivative needed by GammaPoisson.

    JAX does not provide derivatives with respect to the first two arguments of
    ``betainc``. Count is always a detached integer in this prototype, so only
    the first-argument and probability derivatives are needed. The former uses
    a centered finite difference; the latter is the beta density.
    if args.csv_dir is not None:
        print(f"\nCSV files written to {args.csv_dir.resolve()}")

    """
    return betainc(concentration, count, probability)


@_betainc_in_concentration.defjvp
def _betainc_in_concentration_jvp(primals, tangents):
    concentration, count, probability = primals
    concentration_dot, _count_dot, probability_dot = tangents
    dtype = jnp.result_type(concentration, probability)
    relative_step = jnp.finfo(dtype).eps ** (1.0 / 3.0)
    step = relative_step * jnp.maximum(concentration, 1.0)
    lower = jnp.maximum(concentration - step, jnp.finfo(dtype).tiny)
    upper = concentration + step
    value = betainc(concentration, count, probability)
    concentration_derivative = (
        betainc(upper, count, probability) - betainc(lower, count, probability)
    ) / (upper - lower)
    probability_derivative = jnp.exp(
        (concentration - 1.0) * jnp.log(probability)
        + (count - 1.0) * jnp.log1p(-probability)
        - betaln(concentration, count)
    )
    tangent = (
        concentration_derivative * concentration_dot
        + probability_derivative * probability_dot
    )
    return value, tangent


def _expand_parameter(value, target):
    """Append sample axes so a batch parameter broadcasts against ``target``."""
    value = jnp.asarray(value)
    return jnp.reshape(value, jnp.shape(value) + (1,) * (jnp.ndim(target) - value.ndim))


def _count_cdf(base_dist, value):
    """CDF shared by all unbounded families accepted by ``adaptive_relaxed_count``."""
    value = jnp.floor(value)

    if isinstance(base_dist, dist.Poisson):
        rate = _expand_parameter(base_dist.rate, value)
        cdf = gammaincc(value + 1.0, rate)
    elif isinstance(base_dist, (dist.GeometricProbs, dist.GeometricLogits)):
        probs = _expand_parameter(base_dist.probs, value)
        cdf = -jnp.expm1((value + 1.0) * jnp.log1p(-probs))
    elif isinstance(base_dist, dist.GammaPoisson):
        concentration = _expand_parameter(base_dist.concentration, value)
        rate = _expand_parameter(base_dist.rate, value)
        cdf = _betainc_in_concentration(concentration, value + 1.0, rate / (rate + 1.0))
    elif isinstance(base_dist, dist.ZeroInflatedProbs):
        gate = _expand_parameter(base_dist.gate, value)
        cdf = gate + (1.0 - gate) * _count_cdf(base_dist.base_dist, value)
    else:
        raise ValueError(
            f"anchored relaxed count does not support {type(base_dist).__name__}"
        )

    return jnp.where(value < 0, 0.0, cdf)


def _count_log_pmf(base_dist, value):
    """Parameter-explicit log PMF with reliable broadcasting over window axes."""
    if isinstance(base_dist, dist.Poisson):
        rate = _expand_parameter(base_dist.rate, value)
        return value * jnp.log(rate) - rate - gammaln(value + 1.0)
    if isinstance(base_dist, (dist.GeometricProbs, dist.GeometricLogits)):
        probs = _expand_parameter(base_dist.probs, value)
        return jnp.log(probs) + value * jnp.log1p(-probs)
    if isinstance(base_dist, dist.GammaPoisson):
        concentration = _expand_parameter(base_dist.concentration, value)
        rate = _expand_parameter(base_dist.rate, value)
        return (
            gammaln(value + concentration)
            - gammaln(value + 1.0)
            - gammaln(concentration)
            + concentration * (jnp.log(rate) - jnp.log1p(rate))
            - value * jnp.log1p(rate)
        )
    if isinstance(base_dist, dist.ZeroInflatedProbs):
        gate = _expand_parameter(base_dist.gate, value)
        base_log_pmf = _count_log_pmf(base_dist.base_dist, value)
        ordinary = jnp.log1p(-gate) + base_log_pmf
        zero = jnp.logaddexp(jnp.log(gate), ordinary)
        return jnp.where(value == 0, zero, ordinary)
    raise ValueError(
        f"anchored relaxed count does not support {type(base_dist).__name__}"
    )


def _pmf_ratio_down(base_dist, value):
    r"""Return :math:`p(k-1) / p(k)` at integer ``value`` :math:`k`."""
    if isinstance(base_dist, dist.Poisson):
        rate = _expand_parameter(base_dist.rate, value)
        return value / rate
    if isinstance(base_dist, (dist.GeometricProbs, dist.GeometricLogits)):
        probs = _expand_parameter(base_dist.probs, value)
        return jnp.ones_like(value) / (1.0 - probs)
    if isinstance(base_dist, dist.GammaPoisson):
        concentration = _expand_parameter(base_dist.concentration, value)
        rate = _expand_parameter(base_dist.rate, value)
        return value * (rate + 1.0) / (value + concentration - 1.0)
    if isinstance(base_dist, dist.ZeroInflatedProbs):
        base_ratio = _pmf_ratio_down(base_dist.base_dist, value)
        # Only the 1 -> 0 transition differs from the base recurrence.
        log_zero_ratio = _count_log_pmf(
            base_dist, jnp.zeros_like(value)
        ) - _count_log_pmf(base_dist, jnp.ones_like(value))
        # Inactive ``where`` branches are evaluated and differentiated too.
        max_log = jnp.log(jnp.finfo(log_zero_ratio.dtype).max) - 2.0
        zero_ratio = jnp.exp(jnp.minimum(log_zero_ratio, max_log))
        return jnp.where(value == 1, zero_ratio, base_ratio)
    raise ValueError(
        f"anchored relaxed count does not support {type(base_dist).__name__}"
    )


def _count_skewness(base_dist):
    """Skewness used by the first-order Cornish--Fisher anchor."""
    if isinstance(base_dist, dist.Poisson):
        return 1.0 / jnp.sqrt(base_dist.rate)
    if isinstance(base_dist, (dist.GeometricProbs, dist.GeometricLogits)):
        probs = base_dist.probs
        return (2.0 - probs) / jnp.sqrt(1.0 - probs)
    if isinstance(base_dist, dist.GammaPoisson):
        concentration = base_dist.concentration
        rate = base_dist.rate
        return (rate + 2.0) / jnp.sqrt(concentration * (rate + 1.0))
    if isinstance(base_dist, dist.ZeroInflatedProbs):
        keep = 1.0 - base_dist.gate
        mean = base_dist.base_dist.mean
        variance = base_dist.base_dist.variance
        third_central = _count_skewness(base_dist.base_dist) * variance**1.5
        raw_second = variance + mean**2
        raw_third = third_central + 3.0 * mean * variance + mean**3
        mixed_mean = keep * mean
        mixed_second = keep * raw_second
        mixed_third = keep * raw_third
        mixed_central = (
            mixed_third - 3.0 * mixed_mean * mixed_second + 2.0 * mixed_mean**3
        )
        return mixed_central / base_dist.variance**1.5
    raise ValueError(
        f"anchored relaxed count does not support {type(base_dist).__name__}"
    )


def count_anchor_binary_search(base_dist, u, max_count=DEFAULT_MAX_COUNT):
    """Return a lower quantile using a fixed-depth search over a count CDF.

    ``base_dist`` may be any unbounded family supported by
    :func:`numpyro.contrib.diag_sgd.adaptive_relaxed_count`. The result is
    clipped at ``max_count`` if that value does not bracket the requested
    quantile, and is detached because an integer anchor must not carry a
    derivative.
    """
    shape = jnp.broadcast_shapes(base_dist.batch_shape, jnp.shape(u))
    lower = jnp.zeros(shape, dtype=int)
    upper = jnp.full(shape, max_count, dtype=int)
    for _ in range(math.ceil(math.log2(max_count + 1))):
        middle = (lower + upper) // 2
        middle_cdf = _count_cdf(base_dist, middle)
        lower = jnp.where(middle_cdf < u, middle + 1, lower)
        upper = jnp.where(middle_cdf < u, upper, middle)
    return jax.lax.stop_gradient(lower)


def count_anchor_cornish_fisher(base_dist, u):
    """Approximate a count quantile from its mean, variance, and skewness."""
    dtype = jnp.result_type(base_dist.mean, u)
    epsilon = jnp.finfo(dtype).eps
    normal_quantile = ndtri(jnp.clip(u, epsilon, 1.0 - epsilon))
    scale = jnp.sqrt(base_dist.variance)
    varying_quantile = base_dist.mean + scale * normal_quantile
    varying_quantile += (
        _count_skewness(base_dist) * scale * (jnp.square(normal_quantile) - 1.0) / 6.0
    )
    # Cornish--Fisher is undefined for a deterministic distribution.
    quantile = jnp.where(scale > 0.0, varying_quantile, base_dist.mean)
    anchor = jnp.maximum(jnp.floor(quantile), 0).astype(int)
    return jax.lax.stop_gradient(anchor)


def anchored_relaxed_count_direct(base_dist, anchor, eta, u, width):
    """Evaluate the fixed-window anchored identity with direct CDF calls."""
    offsets = jnp.arange(-width, width)
    indices = anchor[..., None] + offsets
    valid = indices >= 0
    cdf = _count_cdf(base_dist, jnp.maximum(indices, 0))
    a = jax.nn.sigmoid((u[..., None] - cdf) / eta)
    a_infinity = jax.nn.sigmoid((u - 1.0) / eta)
    a_minus_one = jax.nn.sigmoid(u / eta)
    corrections = jnp.where(
        indices < anchor[..., None],
        a - a_minus_one[..., None],
        a - a_infinity[..., None],
    )
    corrections = jnp.where(valid, corrections, 0.0)
    return anchor + jnp.sum(corrections, axis=-1) / (a_minus_one - a_infinity)


def anchored_relaxed_count_recurrence(base_dist, anchor, eta, u, width):
    """Evaluate the anchored identity from one CDF and local PMF recurrences."""
    dtype = jnp.result_type(base_dist.mean, u)
    anchor_float = anchor.astype(dtype)
    anchor_cdf = _count_cdf(base_dist, anchor_float)
    anchor_pmf = jnp.exp(_count_log_pmf(base_dist, anchor_float))
    steps = jnp.arange(1, width + 1, dtype=dtype)

    left_values = anchor_float[..., None] - steps + 1.0
    left_valid = steps <= anchor_float[..., None]
    left_factors = jnp.where(left_valid, _pmf_ratio_down(base_dist, left_values), 1.0)
    left_pmf = anchor_pmf[..., None] * jnp.cumprod(left_factors, axis=-1)
    left_pmf = jnp.where(left_valid, left_pmf, 0.0)
    left_cdf = anchor_cdf[..., None] - jnp.cumsum(
        jnp.concatenate([anchor_pmf[..., None], left_pmf[..., :-1]], axis=-1),
        axis=-1,
    )
    left_cdf = jnp.flip(left_cdf, axis=-1)

    right_values = anchor_float[..., None] + steps
    right_factors = 1.0 / _pmf_ratio_down(base_dist, right_values)
    right_pmf = anchor_pmf[..., None] * jnp.cumprod(right_factors, axis=-1)
    right_cdf = anchor_cdf[..., None] + jnp.cumsum(right_pmf[..., :-1], axis=-1)
    cdf = jnp.concatenate([left_cdf, anchor_cdf[..., None], right_cdf], axis=-1)
    cdf = jnp.clip(cdf, 0.0, 1.0)

    offsets = jnp.arange(-width, width)
    indices = anchor[..., None] + offsets
    valid = indices >= 0
    a = jax.nn.sigmoid((u[..., None] - cdf) / eta)
    a_infinity = jax.nn.sigmoid((u - 1.0) / eta)
    a_minus_one = jax.nn.sigmoid(u / eta)
    corrections = jnp.where(
        indices < anchor[..., None],
        a - a_minus_one[..., None],
        a - a_infinity[..., None],
    )
    corrections = jnp.where(valid, corrections, 0.0)
    return anchor + jnp.sum(corrections, axis=-1) / (a_minus_one - a_infinity)


def poisson_anchor_binary_search(rate, u, max_count=DEFAULT_MAX_COUNT):
    """Return the lower Poisson quantile using a fixed-depth CDF search."""
    dtype = jnp.result_type(rate)
    lower = jnp.zeros(jnp.broadcast_shapes(jnp.shape(rate), jnp.shape(u)), dtype=int)
    upper = jnp.full_like(lower, max_count)
    for _ in range(math.ceil(math.log2(max_count + 1))):
        middle = (lower + upper) // 2
        middle_cdf = gammaincc(middle.astype(dtype) + 1.0, rate)
        lower = jnp.where(middle_cdf < u, middle + 1, lower)
        upper = jnp.where(middle_cdf < u, upper, middle)
    return jax.lax.stop_gradient(lower)


def poisson_anchor_cornish_fisher(rate, u):
    r"""Approximate the Poisson quantile with a skewness correction.

    For :math:`s=\Phi^{-1}(u)`, the first Cornish--Fisher correction gives

    .. math::

        m \approx \left\lfloor
            \lambda + \sqrt{\lambda}s + \frac{s^2 - 1}{6}
        \right\rfloor.
    """
    dtype = jnp.result_type(rate)
    epsilon = jnp.finfo(dtype).eps
    normal_quantile = ndtri(jnp.clip(u, epsilon, 1.0 - epsilon))
    quantile = rate + jnp.sqrt(rate) * normal_quantile
    quantile = quantile + (jnp.square(normal_quantile) - 1.0) / 6.0
    anchor = jnp.maximum(jnp.floor(quantile), 0).astype(int)
    return jax.lax.stop_gradient(anchor)


def poisson_anchored_relaxed_count_direct(anchor, eta, rate, u, width):
    r"""Evaluate the anchored identity using a direct CDF at every index.

    For :math:`D=a_{-1}-a_\infty`, the exact identity at any integer anchor
    :math:`m \geq 0` is

    .. math::

        z_\eta(u) = m + \frac{
            \sum_{k < m}(a_k-a_{-1}) +
            \sum_{k \geq m}(a_k-a_\infty)
        }{D}.

    This prototype retains only ``width`` indices on each side of ``anchor``.
    """
    dtype = jnp.result_type(rate)
    offsets = jnp.arange(-width, width)
    indices = anchor[..., None] + offsets
    valid = indices >= 0
    safe_indices = jnp.maximum(indices, 0).astype(dtype)
    cdf = gammaincc(safe_indices + 1.0, rate[..., None])
    a = jax.nn.sigmoid((u[..., None] - cdf) / eta)
    a_infinity = jax.nn.sigmoid((u - 1.0) / eta)
    a_minus_one = jax.nn.sigmoid(u / eta)
    corrections = jnp.where(
        indices < anchor[..., None],
        a - a_minus_one[..., None],
        a - a_infinity[..., None],
    )
    corrections = jnp.where(valid, corrections, 0.0)
    denominator = a_minus_one - a_infinity
    return anchor + jnp.sum(corrections, axis=-1) / denominator


def poisson_anchored_relaxed_count_recurrence(anchor, eta, rate, u, width):
    r"""Evaluate the anchored identity from a CDF and a local pmf recurrence.

    The CDF at the anchor comes from ``gammaincc``. Its rate derivative gives
    the anchor pmf exactly, since ``d F_rate(k) / d rate = -p_rate(k)``. The
    other pmfs follow from ``p(k-1) = p(k) k / rate`` and
    ``p(k+1) = p(k) rate / (k+1)``. The CDF values are then cumulative sums
    inside the fixed window, so the number of special-function evaluations is
    independent of ``width``.
    """
    dtype = jnp.result_type(rate)
    offsets = jnp.arange(-width, width)
    indices = anchor[..., None] + offsets
    valid = indices >= 0
    anchor_float = anchor.astype(dtype)
    anchor_cdf = gammaincc(anchor_float + 1.0, rate)
    _, anchor_cdf_tangent = jax.jvp(
        lambda value: gammaincc(anchor_float + 1.0, value),
        (rate,),
        (jnp.ones_like(rate),),
    )
    anchor_pmf = -anchor_cdf_tangent
    steps = jnp.arange(1, width + 1, dtype=dtype)
    left_valid = steps <= anchor_float[..., None]
    left_factors = jnp.where(
        left_valid,
        (anchor_float[..., None] - steps + 1.0) / rate[..., None],
        1.0,
    )
    left_pmf = anchor_pmf[..., None] * jnp.cumprod(left_factors, axis=-1)
    left_pmf = jnp.where(left_valid, left_pmf, 0.0)
    left_cdf = anchor_cdf[..., None] - jnp.cumsum(
        jnp.concatenate([anchor_pmf[..., None], left_pmf[..., :-1]], axis=-1),
        axis=-1,
    )
    left_cdf = jnp.flip(left_cdf, axis=-1)
    right_factors = rate[..., None] / (anchor_float[..., None] + steps)
    right_pmf = anchor_pmf[..., None] * jnp.cumprod(right_factors, axis=-1)
    right_cdf = anchor_cdf[..., None] + jnp.cumsum(right_pmf[..., :-1], axis=-1)
    cdf = jnp.concatenate([left_cdf, anchor_cdf[..., None], right_cdf], axis=-1)
    cdf = jnp.clip(cdf, 0.0, 1.0)
    a = jax.nn.sigmoid((u[..., None] - cdf) / eta)
    a_infinity = jax.nn.sigmoid((u - 1.0) / eta)
    a_minus_one = jax.nn.sigmoid(u / eta)
    corrections = jnp.where(
        indices < anchor[..., None],
        a - a_minus_one[..., None],
        a - a_infinity[..., None],
    )
    corrections = jnp.where(valid, corrections, 0.0)
    denominator = a_minus_one - a_infinity
    return anchor + jnp.sum(corrections, axis=-1) / denominator


if __name__ == "__main__":
    main()
