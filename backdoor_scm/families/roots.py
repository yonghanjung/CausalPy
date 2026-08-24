"""Frozen independent root distributions for active-v1 covariates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

import numpy as np
from scipy.special import betainc, betaincinv, ndtr, ndtri, stdtr, stdtrit

from ..schema import VariableSpec, VariableType
from ..specs import (
    FrozenValue,
    canonical_hash,
    canonical_json,
    freeze_mapping,
    thaw_mapping,
)


def _require_exact_keys(parameters: Mapping[str, Any], keys: set[str]) -> None:
    if set(parameters) != keys:
        raise ValueError(f"Root parameters must have exactly these keys: {sorted(keys)}.")


def _require_finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Root parameter {name!r} must be finite.")
    return result


def _validate_root_spec(spec: "RootSpec") -> None:
    if spec.version != "1.0.0":
        raise ValueError("Unsupported root family version.")
    parameters = spec.parameter_dict()
    continuous_families = {
        "bdpfn.root.normal",
        "bdpfn.root.uniform",
        "bdpfn.root.laplace_standardized",
        "bdpfn.root.beta_standardized",
        "bdpfn.root.gaussian_mixture2_standardized",
        "bdpfn.root.student_t_standardized",
    }
    finite_families = {
        "bdpfn.root.bernoulli",
        "bdpfn.root.categorical_uniform",
        "bdpfn.root.categorical_dirichlet1",
        "bdpfn.root.categorical_truncated_zipf",
        "bdpfn.root.ordinal_randint",
        "bdpfn.root.ordinal_truncated_zipf",
    }
    if spec.family_id in finite_families:
        if spec.family_id == "bdpfn.root.bernoulli":
            if spec.variable_type is not VariableType.BINARY or spec.cardinality != 2:
                raise ValueError("Bernoulli root requires a binary variable.")
            _require_exact_keys(parameters, {"p", "probability_mode"})
            probability = _require_finite(parameters["p"], "p")
            mode = str(parameters["probability_mode"])
            if mode == "point_mass_0.5":
                if probability != 0.5:
                    raise ValueError("Point-mass Bernoulli mode requires p=0.5.")
            elif mode == "uniform_0.1_0.9":
                if not 0.1 <= probability <= 0.9:
                    raise ValueError("Uniform Bernoulli draw must lie in [0.1, 0.9].")
            else:
                raise ValueError("Unknown Bernoulli probability mode.")
            return
        if spec.cardinality is None:
            raise ValueError("Finite root requires a cardinality.")
        if spec.family_id.startswith("bdpfn.root.categorical_"):
            if spec.variable_type is not VariableType.CATEGORICAL:
                raise ValueError("Categorical root requires a categorical variable.")
            keys = {"base_pmf", "pmf"}
            if spec.family_id.endswith("truncated_zipf"):
                keys.add("exponent")
            _require_exact_keys(parameters, keys)
            base = np.asarray(parameters["base_pmf"], dtype=float)
            probability = np.asarray(parameters["pmf"], dtype=float)
            if base.shape != (spec.cardinality,) or probability.shape != (spec.cardinality,):
                raise ValueError("Categorical PMF has the wrong cardinality.")
            if not np.isfinite(base).all() or (base <= 0.0).any() or not math.isclose(
                float(base.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("Categorical base PMF is invalid.")
            expected = 0.9 * base + 0.1 / spec.cardinality
            if not np.allclose(probability, expected, rtol=0.0, atol=1e-12):
                raise ValueError("Categorical PMF does not satisfy the floor mixture.")
            if spec.family_id.endswith("uniform") and not np.allclose(
                base, np.full(spec.cardinality, 1.0 / spec.cardinality), rtol=0.0, atol=1e-12
            ):
                raise ValueError("Uniform categorical base PMF is not uniform.")
            if spec.family_id.endswith("truncated_zipf"):
                exponent = _require_finite(parameters["exponent"], "exponent")
                if not 0.5 <= exponent <= 2.0:
                    raise ValueError("Zipf exponent must lie in [0.5, 2].")
                expected_base = np.arange(1, spec.cardinality + 1, dtype=float) ** (-exponent)
                expected_base /= expected_base.sum()
                if not np.allclose(base, expected_base, rtol=0.0, atol=1e-12):
                    raise ValueError("Categorical Zipf base PMF is inconsistent.")
            return
        if spec.variable_type is not VariableType.ORDINAL:
            raise ValueError("Ordinal root requires an ordinal variable.")
        keys = {"pmf"}
        if spec.family_id.endswith("truncated_zipf"):
            keys.add("exponent")
        _require_exact_keys(parameters, keys)
        probability = np.asarray(parameters["pmf"], dtype=float)
        if probability.shape != (spec.cardinality,) or not np.isfinite(probability).all():
            raise ValueError("Ordinal PMF has the wrong shape or nonfinite values.")
        if spec.family_id.endswith("randint"):
            expected = np.full(spec.cardinality, 1.0 / spec.cardinality)
        else:
            exponent = _require_finite(parameters["exponent"], "exponent")
            if not 0.5 <= exponent <= 2.0:
                raise ValueError("Ordinal Zipf exponent must lie in [0.5, 2].")
            expected = np.arange(1, spec.cardinality + 1, dtype=float) ** (-exponent)
            expected /= expected.sum()
        if not np.allclose(probability, expected, rtol=0.0, atol=1e-12):
            raise ValueError("Ordinal PMF is inconsistent with its family.")
        return
    if spec.family_id not in continuous_families:
        raise ValueError(f"Unsupported root family: {spec.family_id!r}")
    if spec.variable_type is not VariableType.CONTINUOUS or spec.cardinality is not None:
        raise ValueError("Continuous root family has an incompatible variable type.")
    if spec.family_id == "bdpfn.root.normal":
        _require_exact_keys(parameters, set())
    elif spec.family_id == "bdpfn.root.uniform":
        _require_exact_keys(parameters, {"low", "high"})
        low = _require_finite(parameters["low"], "low")
        high = _require_finite(parameters["high"], "high")
        if not math.isclose(low, -math.sqrt(3.0), rel_tol=0.0, abs_tol=1e-15) or not math.isclose(
            high, math.sqrt(3.0), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("Active-v1 Uniform root must be exactly standardized.")
    elif spec.family_id == "bdpfn.root.laplace_standardized":
        _require_exact_keys(parameters, {"location", "scale"})
        location = _require_finite(parameters["location"], "location")
        scale = _require_finite(parameters["scale"], "scale")
        if location != 0.0 or not math.isclose(
            scale, 1.0 / math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("Active-v1 Laplace root must be exactly standardized.")
    elif spec.family_id == "bdpfn.root.beta_standardized":
        _require_exact_keys(parameters, {"alpha", "beta", "raw_mean", "raw_sd"})
        alpha = _require_finite(parameters["alpha"], "alpha")
        beta = _require_finite(parameters["beta"], "beta")
        raw_mean = _require_finite(parameters["raw_mean"], "raw_mean")
        raw_sd = _require_finite(parameters["raw_sd"], "raw_sd")
        if not 0.5 <= alpha <= 5.0 or not 0.5 <= beta <= 5.0:
            raise ValueError("Beta shape parameters must lie in [0.5, 5].")
        expected_mean = alpha / (alpha + beta)
        expected_variance = alpha * beta / (
            (alpha + beta) ** 2 * (alpha + beta + 1.0)
        )
        if raw_sd <= 0.0 or not math.isclose(
            raw_mean, expected_mean, rel_tol=1e-14, abs_tol=1e-14
        ) or not math.isclose(
            raw_sd, math.sqrt(expected_variance), rel_tol=1e-14, abs_tol=1e-14
        ):
            raise ValueError("Beta standardization parameters are inconsistent.")
    elif spec.family_id == "bdpfn.root.gaussian_mixture2_standardized":
        _require_exact_keys(
            parameters,
            {"weight", "separation", "sd0", "sd1", "raw_mean", "raw_sd"},
        )
        weight = _require_finite(parameters["weight"], "weight")
        separation = _require_finite(parameters["separation"], "separation")
        sd0 = _require_finite(parameters["sd0"], "sd0")
        sd1 = _require_finite(parameters["sd1"], "sd1")
        raw_mean = _require_finite(parameters["raw_mean"], "raw_mean")
        raw_sd = _require_finite(parameters["raw_sd"], "raw_sd")
        if not 0.2 <= weight <= 0.8 or not 0.5 <= separation <= 3.0:
            raise ValueError("Gaussian-mixture weight or separation is out of range.")
        if not 0.2 <= sd0 <= 1.5 or not 0.2 <= sd1 <= 1.5:
            raise ValueError("Gaussian-mixture component SD is out of range.")
        mean0, mean1 = -separation / 2.0, separation / 2.0
        expected_mean = weight * mean0 + (1.0 - weight) * mean1
        second = weight * (sd0**2 + mean0**2) + (1.0 - weight) * (
            sd1**2 + mean1**2
        )
        expected_sd = math.sqrt(second - expected_mean**2)
        if raw_sd <= 0.0 or not math.isclose(
            raw_mean, expected_mean, rel_tol=1e-14, abs_tol=1e-14
        ) or not math.isclose(
            raw_sd, expected_sd, rel_tol=1e-14, abs_tol=1e-14
        ):
            raise ValueError("Gaussian-mixture standardization is inconsistent.")
    else:
        _require_exact_keys(parameters, {"df", "scale"})
        df = _require_finite(parameters["df"], "df")
        scale = _require_finite(parameters["scale"], "scale")
        expected_scale = math.sqrt((df - 2.0) / df)
        if not 3.0 <= df <= 32.0 or not math.isclose(
            scale, expected_scale, rel_tol=1e-14, abs_tol=1e-14
        ):
            raise ValueError("Student-t df or variance standardization is inconsistent.")


@dataclass(frozen=True)
class RootSpec:
    family_id: str
    version: str
    variable_type: VariableType
    cardinality: int | None
    parameters: tuple[tuple[str, FrozenValue], ...]

    def __post_init__(self) -> None:
        if not self.family_id or not self.version:
            raise ValueError("Root family ID and version must be non-empty.")
        variable = VariableSpec("root", self.variable_type, self.cardinality)
        object.__setattr__(self, "variable_type", variable.variable_type)
        _validate_root_spec(self)

    @classmethod
    def create(
        cls,
        family_id: str,
        version: str,
        variable_type: VariableType,
        cardinality: int | None,
        parameters: Mapping[str, Any],
    ) -> "RootSpec":
        return cls(
            family_id=str(family_id),
            version=str(version),
            variable_type=VariableType(variable_type),
            cardinality=cardinality,
            parameters=freeze_mapping(parameters),
        )

    def parameter_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "version": self.version,
            "variable_type": self.variable_type.value,
            "cardinality": self.cardinality,
            "parameters": self.parameter_dict(),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RootSpec":
        return cls.create(
            family_id=str(data["family_id"]),
            version=str(data["version"]),
            variable_type=VariableType(str(data["variable_type"])),
            cardinality=(
                None if data.get("cardinality") is None else int(data["cardinality"])
            ),
            parameters=data["parameters"],
        )

    @classmethod
    def from_json(cls, encoded: str) -> "RootSpec":
        return cls.from_dict(json.loads(encoded))


@dataclass(frozen=True)
class RootLaw:
    spec: RootSpec

    @property
    def mean(self) -> float:
        if self.spec.variable_type is VariableType.CONTINUOUS:
            return 0.0
        parameters = self.spec.parameter_dict()
        if self.spec.variable_type is VariableType.BINARY:
            return float(parameters["p"])
        probabilities = np.asarray(parameters["pmf"], dtype=float)
        return float(np.arange(probabilities.size) @ probabilities)

    @property
    def variance(self) -> float:
        if self.spec.variable_type is VariableType.CONTINUOUS:
            return 1.0
        parameters = self.spec.parameter_dict()
        if self.spec.variable_type is VariableType.BINARY:
            probability = float(parameters["p"])
            return probability * (1.0 - probability)
        probabilities = np.asarray(parameters["pmf"], dtype=float)
        support = np.arange(probabilities.size, dtype=float)
        mean = float(support @ probabilities)
        return float((support * support) @ probabilities - mean * mean)

    @property
    def pmf(self) -> np.ndarray | None:
        if self.spec.variable_type is VariableType.CONTINUOUS:
            return None
        parameters = self.spec.parameter_dict()
        if self.spec.variable_type is VariableType.BINARY:
            probability = float(parameters["p"])
            result = np.asarray([1.0 - probability, probability])
        else:
            result = np.asarray(parameters["pmf"], dtype=float)
        result.setflags(write=False)
        return result

    def contains(self, values: Any) -> bool:
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if numeric.ndim != 1 or not np.isfinite(numeric).all():
            return False
        parameters = self.spec.parameter_dict()
        if self.spec.variable_type is VariableType.CONTINUOUS:
            if self.spec.family_id == "bdpfn.root.uniform":
                return bool(
                    (numeric >= parameters["low"]).all()
                    and (numeric <= parameters["high"]).all()
                )
            if self.spec.family_id == "bdpfn.root.beta_standardized":
                lower = (0.0 - parameters["raw_mean"]) / parameters["raw_sd"]
                upper = (1.0 - parameters["raw_mean"]) / parameters["raw_sd"]
                return bool((numeric >= lower).all() and (numeric <= upper).all())
            return True
        if not np.equal(numeric, np.floor(numeric)).all():
            return False
        return bool((numeric >= 0.0).all() and (numeric < self.spec.cardinality).all())

    def cdf(self, values: Any) -> np.ndarray:
        if self.spec.variable_type is not VariableType.CONTINUOUS:
            raise ValueError("CDF transform is defined only for continuous roots.")
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("CDF inputs must be numeric.") from error
        if not np.isfinite(numeric).all():
            raise ValueError("CDF inputs must be finite.")
        parameters = self.spec.parameter_dict()
        if self.spec.family_id == "bdpfn.root.normal":
            result = ndtr(numeric)
        elif self.spec.family_id == "bdpfn.root.uniform":
            result = np.clip(
                (numeric - parameters["low"])
                / (parameters["high"] - parameters["low"]),
                0.0,
                1.0,
            )
        elif self.spec.family_id == "bdpfn.root.laplace_standardized":
            centered = (numeric - parameters["location"]) / parameters["scale"]
            result = np.empty_like(centered)
            lower = centered < 0.0
            result[lower] = 0.5 * np.exp(centered[lower])
            result[~lower] = 1.0 - 0.5 * np.exp(-centered[~lower])
        elif self.spec.family_id == "bdpfn.root.beta_standardized":
            raw = numeric * parameters["raw_sd"] + parameters["raw_mean"]
            result = betainc(
                parameters["alpha"],
                parameters["beta"],
                np.clip(raw, 0.0, 1.0),
            )
            result = np.where(raw <= 0.0, 0.0, np.where(raw >= 1.0, 1.0, result))
        elif self.spec.family_id == "bdpfn.root.gaussian_mixture2_standardized":
            raw = numeric * parameters["raw_sd"] + parameters["raw_mean"]
            mean0 = -parameters["separation"] / 2.0
            mean1 = parameters["separation"] / 2.0
            result = parameters["weight"] * ndtr(
                (raw - mean0) / parameters["sd0"]
            ) + (1.0 - parameters["weight"]) * ndtr(
                (raw - mean1) / parameters["sd1"]
            )
        else:
            result = stdtr(parameters["df"], numeric / parameters["scale"])
        frozen = np.array(result, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        return frozen

    def ppf(self, probabilities: Any) -> np.ndarray:
        try:
            probability = np.asarray(probabilities, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Quantile probabilities must be numeric.") from error
        if (
            not np.isfinite(probability).all()
            or (probability < 0.0).any()
            or (probability > 1.0).any()
        ):
            raise ValueError("Quantile probabilities must lie in [0, 1].")
        parameters = self.spec.parameter_dict()
        if self.spec.variable_type is not VariableType.CONTINUOUS:
            cumulative = np.cumsum(self.pmf)
            result = np.searchsorted(cumulative, probability, side="left")
            result = np.minimum(result, self.spec.cardinality - 1).astype(np.int64)
            result.setflags(write=False)
            return result
        if self.spec.family_id == "bdpfn.root.normal":
            result = ndtri(probability)
        elif self.spec.family_id == "bdpfn.root.uniform":
            result = parameters["low"] + probability * (
                parameters["high"] - parameters["low"]
            )
        elif self.spec.family_id == "bdpfn.root.laplace_standardized":
            result = np.empty_like(probability)
            lower = probability < 0.5
            with np.errstate(divide="ignore"):
                result[lower] = parameters["location"] + parameters["scale"] * np.log(
                    2.0 * probability[lower]
                )
                result[~lower] = parameters["location"] - parameters["scale"] * np.log(
                    2.0 * (1.0 - probability[~lower])
                )
        elif self.spec.family_id == "bdpfn.root.beta_standardized":
            raw = betaincinv(
                parameters["alpha"], parameters["beta"], probability
            )
            result = (raw - parameters["raw_mean"]) / parameters["raw_sd"]
        elif self.spec.family_id == "bdpfn.root.gaussian_mixture2_standardized":
            result = self._mixture_ppf(probability)
        else:
            result = parameters["scale"] * stdtrit(parameters["df"], probability)
        frozen = np.array(result, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        return frozen

    def _mixture_ppf(self, probability: np.ndarray) -> np.ndarray:
        result = np.empty_like(probability)
        result[probability == 0.0] = -np.inf
        result[probability == 1.0] = np.inf
        interior = (probability > 0.0) & (probability < 1.0)
        if not interior.any():
            return result
        target = probability[interior]
        lower = np.full(target.shape, -1.0)
        upper = np.full(target.shape, 1.0)
        for _ in range(64):
            move_lower = self.cdf(lower) > target
            move_upper = self.cdf(upper) < target
            if not move_lower.any() and not move_upper.any():
                break
            lower = np.where(move_lower, 2.0 * lower - 1.0, lower)
            upper = np.where(move_upper, 2.0 * upper + 1.0, upper)
        else:
            raise ArithmeticError("Gaussian-mixture quantile bracketing did not converge.")
        for _ in range(128):
            midpoint = 0.5 * (lower + upper)
            below = self.cdf(midpoint) < target
            lower = np.where(below, midpoint, lower)
            upper = np.where(below, upper, midpoint)
        result[interior] = 0.5 * (lower + upper)
        return result

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if type(n) is not int or n < 0:
            raise ValueError("Root sample size must be non-negative.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Root sampling requires a local NumPy Generator.")
        parameters = self.spec.parameter_dict()
        if self.spec.family_id == "bdpfn.root.normal":
            return rng.normal(size=n)
        if self.spec.family_id == "bdpfn.root.student_t_standardized":
            return parameters["scale"] * rng.standard_t(parameters["df"], size=n)
        if self.spec.family_id == "bdpfn.root.uniform":
            return rng.uniform(parameters["low"], parameters["high"], size=n)
        if self.spec.family_id == "bdpfn.root.laplace_standardized":
            return rng.laplace(
                parameters["location"], parameters["scale"], size=n
            )
        if self.spec.family_id == "bdpfn.root.beta_standardized":
            raw = rng.beta(parameters["alpha"], parameters["beta"], size=n)
            return (raw - parameters["raw_mean"]) / parameters["raw_sd"]
        if self.spec.family_id == "bdpfn.root.gaussian_mixture2_standardized":
            choose_zero = rng.random(n) < parameters["weight"]
            mean0 = -parameters["separation"] / 2.0
            mean1 = parameters["separation"] / 2.0
            means = np.where(choose_zero, mean0, mean1)
            scales = np.where(choose_zero, parameters["sd0"], parameters["sd1"])
            raw = rng.normal(means, scales)
            return (raw - parameters["raw_mean"]) / parameters["raw_sd"]
        if self.spec.variable_type is VariableType.BINARY:
            return (rng.random(n) < parameters["p"]).astype(np.int64)
        if self.spec.variable_type in {VariableType.CATEGORICAL, VariableType.ORDINAL}:
            return rng.choice(
                self.spec.cardinality,
                size=n,
                p=np.asarray(parameters["pmf"], dtype=float),
            ).astype(np.int64)
        raise ValueError(f"Unsupported root family: {self.spec.family_id!r}")


def build_root(spec: RootSpec) -> RootLaw:
    if not isinstance(spec, RootSpec):
        raise TypeError("Root build input must be a RootSpec.")
    return RootLaw(spec)


_ROOT_FAMILY_WEIGHTS: dict[VariableType, tuple[tuple[str, float], ...]] = {
    VariableType.CONTINUOUS: (
        ("normal", 0.20),
        ("uniform", 0.15),
        ("beta", 0.10),
        ("laplace", 0.15),
        ("gaussian_mixture2", 0.15),
        ("student_t", 0.25),
    ),
    VariableType.BINARY: (("bernoulli", 1.0),),
    VariableType.CATEGORICAL: (
        ("uniform", 0.35),
        ("dirichlet1", 0.35),
        ("truncated_zipf", 0.30),
    ),
    VariableType.ORDINAL: (("randint", 0.50), ("truncated_zipf", 0.50)),
}


def root_family_weights(variable_type: VariableType) -> dict[str, float]:
    """Return a fresh copy of the active-v1 compatible-family law."""

    try:
        normalized = VariableType(variable_type)
    except ValueError as error:
        raise ValueError("Unknown root variable type.") from error
    return dict(_ROOT_FAMILY_WEIGHTS[normalized])


def _choose_label(
    weighted_labels: tuple[tuple[str, float], ...],
    rng: np.random.Generator,
) -> str:
    probabilities = np.asarray([weight for _, weight in weighted_labels], dtype=float)
    probabilities /= probabilities.sum()
    return weighted_labels[int(rng.choice(len(weighted_labels), p=probabilities))][0]


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_root_spec(
    variable: VariableSpec,
    rng: np.random.Generator,
) -> RootSpec:
    """Draw and freeze one active-v1 root law using only ``rng``."""

    if not isinstance(variable, VariableSpec):
        raise TypeError("Root specification sampling requires a VariableSpec.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Root specification sampling requires a local NumPy Generator.")
    label = _choose_label(_ROOT_FAMILY_WEIGHTS[variable.variable_type], rng)
    family_id = f"bdpfn.root.{label}"
    parameters: dict[str, Any]
    if variable.variable_type is VariableType.CONTINUOUS:
        if label == "normal":
            parameters = {}
        elif label == "uniform":
            parameters = {"low": -math.sqrt(3.0), "high": math.sqrt(3.0)}
        elif label == "laplace":
            family_id = "bdpfn.root.laplace_standardized"
            parameters = {"location": 0.0, "scale": 1.0 / math.sqrt(2.0)}
        elif label == "beta":
            family_id = "bdpfn.root.beta_standardized"
            alpha = _loguniform(rng, 0.5, 5.0)
            beta = _loguniform(rng, 0.5, 5.0)
            raw_mean = alpha / (alpha + beta)
            raw_variance = alpha * beta / (
                (alpha + beta) ** 2 * (alpha + beta + 1.0)
            )
            parameters = {
                "alpha": alpha,
                "beta": beta,
                "raw_mean": raw_mean,
                "raw_sd": math.sqrt(raw_variance),
            }
        elif label == "gaussian_mixture2":
            family_id = "bdpfn.root.gaussian_mixture2_standardized"
            weight = float(rng.uniform(0.2, 0.8))
            separation = float(rng.uniform(0.5, 3.0))
            sd0 = _loguniform(rng, 0.2, 1.5)
            sd1 = _loguniform(rng, 0.2, 1.5)
            mean0, mean1 = -separation / 2.0, separation / 2.0
            raw_mean = weight * mean0 + (1.0 - weight) * mean1
            raw_second = weight * (sd0**2 + mean0**2) + (1.0 - weight) * (
                sd1**2 + mean1**2
            )
            parameters = {
                "weight": weight,
                "separation": separation,
                "sd0": sd0,
                "sd1": sd1,
                "raw_mean": raw_mean,
                "raw_sd": math.sqrt(raw_second - raw_mean**2),
            }
        else:
            family_id = "bdpfn.root.student_t_standardized"
            df = 2.0 + _loguniform(rng, 1.0, 30.0)
            parameters = {"df": df, "scale": math.sqrt((df - 2.0) / df)}
    elif variable.variable_type is VariableType.BINARY:
        mode = _choose_label(
            (("point_mass_0.5", 0.40), ("uniform_0.1_0.9", 0.60)),
            rng,
        )
        probability = 0.5 if mode == "point_mass_0.5" else float(rng.uniform(0.1, 0.9))
        parameters = {"p": probability, "probability_mode": mode}
    elif variable.variable_type is VariableType.CATEGORICAL:
        family_id = f"bdpfn.root.categorical_{label}"
        cardinality = variable.cardinality
        if label == "uniform":
            base = np.full(cardinality, 1.0 / cardinality)
        elif label == "dirichlet1":
            base = rng.dirichlet(np.ones(cardinality))
        else:
            exponent = float(rng.uniform(0.5, 2.0))
            base = np.arange(1, cardinality + 1, dtype=float) ** (-exponent)
            base /= base.sum()
        probability = 0.9 * base + 0.1 / cardinality
        parameters = {
            "base_pmf": base.tolist(),
            "pmf": probability.tolist(),
        }
        if label == "truncated_zipf":
            parameters["exponent"] = exponent
    else:
        family_id = f"bdpfn.root.ordinal_{label}"
        cardinality = variable.cardinality
        if label == "randint":
            probability = np.full(cardinality, 1.0 / cardinality)
        else:
            exponent = float(rng.uniform(0.5, 2.0))
            probability = np.arange(1, cardinality + 1, dtype=float) ** (-exponent)
            probability /= probability.sum()
        parameters = {"pmf": probability.tolist()}
        if label == "truncated_zipf":
            parameters["exponent"] = exponent
    return RootSpec.create(
        family_id=family_id,
        version="1.0.0",
        variable_type=variable.variable_type,
        cardinality=variable.cardinality,
        parameters=parameters,
    )
