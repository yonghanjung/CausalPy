# Canonical random back-door SCM generator

`backdoor_scm` owns immutable synthetic task specifications, exact back-door
truth, and locally keyed row and query sampling. The current built-in manifest
is the continuous P1-P3 smoke slice; additional covariate, propensity, and
outcome families enter only through an explicit `FamilyRegistry`.

## Public task API

- `sample_task(...)` freezes one task from a manifest and registry.
- `load_task(spec_or_json, registry)` reconstructs a frozen semantic task.
- `task.sample_rows(n, row_seed, start_row=0)` returns observed `x`, `a`, `y`.
- `task.sample_query(n, query_seed, start_query=0)` returns a frozen,
  read-only `SemanticQueryBatch`.
- `task.truth(x)`, `task.propensity(x)`, `task.mu(arm, x)`, and `task.tau(x)`
  accept either a numeric two-dimensional NumPy array or a
  `SemanticQueryBatch`. `arm` is integer `0` or `1`.

Wrong rank or covariate dimension raises `ValueError`. A correctly shaped
query outside the covariate law's declared support raises
`OutOfSupportError`.

The full `task_spec_hash` is the serialized-document integrity and provenance
identity. The separate `sampling_identity` contains the task seed lineage,
manifest, component specifications, outcome metadata, and generator RNG
contract, but excludes caller `source_id` and the exact NumPy runtime. Thus a
provenance-only source change does not alter generated rows or queries.

The generator RNG contract and exact NumPy runtime are recorded separately.
Semantic loading and deterministic truth remain available when runtime
provenance differs. Stochastic row and query generation fail closed unless the
RNG algorithm, generator RNG contract, and NumPy runtime all match.

Gaussian and centered-Laplace outcome specifications share `noise_sd`, the
conditional outcome standard deviation. Gaussian sampling uses that standard
deviation directly. Laplace sampling uses native scale
`noise_sd / sqrt(2)` so likelihood choice does not systematically change the
variance prior.

## Test interpreter boundary

From the standalone CausalPy repository root, the core package suite is:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s backdoor_scm/tests -p 'test_*.py' -v
```

Every report must also record `python -VV`, NumPy, and SciPy versions. Passing
this core suite in an arbitrary developer interpreter validates only the
isolated package behavior in that interpreter. It is not full CausalPy
environment certification.

Full validation requires a clean interpreter environment with
`requirements.txt` installed exactly, followed by the core suite and the
maintainer-selected legacy CausalPy test programs. The repository currently
does not pin a Python interpreter version, so that interpreter must be stated
explicitly rather than inferred from the dependency file.
