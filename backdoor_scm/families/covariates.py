"""Built-in covariate laws for the first vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.special import gammaincinv, ndtr, ndtri, stdtr

from ..artifacts import ArtifactResolver, ResolvedArtifact
from ..registry import ComponentRole
from ..schema import CovariateSchema, sample_schema, TypedCovariateBatch
from ..specs import FamilyRef
from .roots import RootSpec, build_root, sample_root_spec


def _dependent_schema_roots(
    schema: CovariateSchema,
    root_specs: tuple[RootSpec, ...],
) -> tuple[RootSpec, ...]:
    roots = tuple(root_specs)
    if schema.dimension < 2:
        raise ValueError("Coordinate dependence requires d>=2.")
    if len(roots) != schema.dimension:
        raise ValueError("Root count disagrees with the continuous schema.")
    for variable, root in zip(schema.variables, roots):
        if (
            root.variable_type is not variable.variable_type
            or root.cardinality != variable.cardinality
        ):
            raise ValueError("Dependent root is incompatible with its variable.")
    return roots


def _matrix(values: Any, dimension: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (dimension, dimension) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite d-by-d matrix.")
    return result


def _normalized_correlation(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    if not np.isfinite(diagonal).all() or (diagonal <= 0.0).any():
        raise ValueError("Covariance diagonal must be positive and finite.")
    scale = np.sqrt(diagonal)
    return covariance / scale[:, None] / scale[None, :]


_DEPENDENCE_STRENGTHS = (
    ("weak", 0.25),
    ("moderate", 0.50),
    ("strong", 0.75),
)


def _sample_dependence_strength(
    rng: np.random.Generator,
) -> tuple[str, float]:
    return _DEPENDENCE_STRENGTHS[int(rng.integers(0, len(_DEPENDENCE_STRENGTHS)))]


def _strengthened_correlation(raw: np.ndarray, alpha: float) -> np.ndarray:
    if alpha not in {value for _, value in _DEPENDENCE_STRENGTHS}:
        raise ValueError("Unknown frozen dependence strength.")
    return (1.0 - alpha) * np.eye(raw.shape[0]) + alpha * raw


def _validate_correlation(values: Any, dimension: int) -> np.ndarray:
    correlation = _matrix(values, dimension, "Correlation")
    if not np.allclose(correlation, correlation.T, rtol=0.0, atol=1e-12):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Correlation matrix must have unit diagonal.")
    try:
        np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError("Correlation matrix must be positive definite.") from error
    return correlation


def _dependent_sample(
    schema: CovariateSchema,
    root_specs: tuple[RootSpec, ...],
    correlation: np.ndarray,
    n: int,
    rng: np.random.Generator,
    student_df: float | None = None,
) -> TypedCovariateBatch:
    if type(n) is not int or n < 0:
        raise ValueError("Covariate sample size must be non-negative.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Covariate sampling requires a local NumPy Generator.")
    cholesky = np.linalg.cholesky(correlation)
    if student_df is None:
        latent = rng.standard_normal((n, schema.dimension)) @ cholesky.T
        probabilities = ndtr(latent)
    else:
        uniforms = rng.random((n, schema.dimension + 1))
        open_lower = np.nextafter(0.0, 1.0)
        open_upper = np.nextafter(1.0, 0.0)
        uniforms = np.clip(uniforms, open_lower, open_upper)
        gaussian = ndtri(uniforms[:, : schema.dimension]) @ cholesky.T
        chi_square = 2.0 * gammaincinv(student_df / 2.0, uniforms[:, -1])
        latent = gaussian / np.sqrt(chi_square[:, None] / student_df)
        probabilities = stdtr(student_df, latent)
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    probabilities = np.where(
        probabilities == 0.0,
        lower,
        np.where(probabilities == 1.0, upper, probabilities),
    )
    return TypedCovariateBatch(
        schema,
        tuple(
            build_root(root).ppf(probabilities[:, index])
            for index, root in enumerate(root_specs)
        ),
    )


def _dependent_contains(
    schema: CovariateSchema,
    root_specs: tuple[RootSpec, ...],
    values: Any,
) -> bool:
    try:
        batch = values if isinstance(values, TypedCovariateBatch) else TypedCovariateBatch.from_matrix(
            schema, np.asarray(values)
        )
    except (TypeError, ValueError):
        return False
    if batch.schema != schema:
        return False
    return all(
        build_root(root).contains(column)
        for root, column in zip(root_specs, batch.columns)
    )


def _schema_roots_from_hyperparameters(
    hyperparameters: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[CovariateSchema, tuple[RootSpec, ...]]:
    if set(hyperparameters) - {"schema", "root_specs", "max_matrix_attempts"}:
        raise ValueError("Dependent-covariate hyperparameters have unexpected fields.")
    if "schema" not in hyperparameters:
        raise ValueError("Dependent-covariate sampling requires a frozen schema.")
    schema = CovariateSchema.from_dict(hyperparameters["schema"])
    if "root_specs" in hyperparameters:
        roots = tuple(RootSpec.from_dict(item) for item in hyperparameters["root_specs"])
    else:
        roots = tuple(sample_root_spec(variable, rng) for variable in schema.variables)
    return schema, _dependent_schema_roots(schema, roots)


@dataclass(frozen=True)
class GaussianCopulaCovariates:
    schema: CovariateSchema
    root_specs: tuple[RootSpec, ...]
    gram_factor: tuple[tuple[float, ...], ...]
    correlation: tuple[tuple[float, ...], ...]
    matrix_sampling_attempt: int
    dependence_strength: str
    strength_alpha: float

    @property
    def dimension(self) -> int:
        return self.schema.dimension

    def __post_init__(self) -> None:
        roots = _dependent_schema_roots(self.schema, self.root_specs)
        if type(self.matrix_sampling_attempt) is not int or self.matrix_sampling_attempt < 0:
            raise ValueError("Matrix sampling attempt must be non-negative.")
        factor = _matrix(self.gram_factor, self.schema.dimension, "Gram factor")
        correlation = _validate_correlation(self.correlation, self.schema.dimension)
        if (self.dependence_strength, self.strength_alpha) not in _DEPENDENCE_STRENGTHS:
            raise ValueError("Dependence strength label and alpha disagree.")
        expected = _strengthened_correlation(
            _normalized_correlation(factor @ factor.T), self.strength_alpha
        )
        if not np.allclose(correlation, expected, rtol=0.0, atol=1e-12):
            raise ValueError("Frozen correlation disagrees with its Gram factor.")
        object.__setattr__(self, "root_specs", roots)
        object.__setattr__(
            self, "gram_factor", tuple(tuple(float(x) for x in row) for row in factor)
        )
        object.__setattr__(
            self, "correlation", tuple(tuple(float(x) for x in row) for row in correlation)
        )

    def sample(self, n: int, rng: np.random.Generator) -> TypedCovariateBatch:
        return _dependent_sample(
            self.schema, self.root_specs, np.asarray(self.correlation), n, rng
        )

    def contains(self, values: Any) -> bool:
        return _dependent_contains(self.schema, self.root_specs, values)


class GaussianCopulaCovariateFactory:
    role = ComponentRole.COVARIATE
    family_id = "bdpfn.covariate.gaussian_copula"
    version = "1.0.0"
    semantic_class_id = "covariate.gaussian_copula"
    outcome_type = None

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Covariate specification sampling requires a local Generator.")
        schema, roots = _schema_roots_from_hyperparameters(hyperparameters, rng)
        strength, alpha = _sample_dependence_strength(rng)
        max_attempts = int(hyperparameters.get("max_matrix_attempts", 100))
        if max_attempts <= 0:
            raise ValueError("Matrix resampling limit must be positive.")
        for attempt in range(max_attempts):
            factor = rng.normal(size=(schema.dimension, schema.dimension))
            correlation = _strengthened_correlation(
                _normalized_correlation(factor @ factor.T), alpha
            )
            try:
                _validate_correlation(correlation, schema.dimension)
            except ValueError:
                continue
            return FamilyRef.create(
                self.role.value,
                self.family_id,
                self.version,
                self.semantic_class_id,
                {
                    "schema": schema.to_dict(),
                    "root_specs": [root.to_dict() for root in roots],
                    "gram_factor": factor.tolist(),
                    "correlation": correlation.tolist(),
                    "matrix_sampling_attempt": attempt,
                    "dependence_strength": strength,
                    "strength_alpha": alpha,
                },
            )
        raise ValueError("Full-rank Gaussian Gram sampling exhausted its retry limit.")

    def build(self, spec: FamilyRef) -> GaussianCopulaCovariates:
        if (
            not isinstance(spec, FamilyRef)
            or spec.role != self.role.value
            or spec.family_id != self.family_id
            or spec.version != self.version
            or spec.semantic_class_id != self.semantic_class_id
        ):
            raise ValueError("FamilyRef does not describe a Gaussian copula.")
        parameters = spec.parameter_dict()
        expected = {
            "schema", "root_specs", "gram_factor", "correlation", "matrix_sampling_attempt",
            "dependence_strength", "strength_alpha"
        }
        if set(parameters) != expected:
            raise ValueError("Gaussian-copula specification has unexpected fields.")
        return GaussianCopulaCovariates(
            CovariateSchema.from_dict(parameters["schema"]),
            tuple(RootSpec.from_dict(item) for item in parameters["root_specs"]),
            tuple(tuple(row) for row in parameters["gram_factor"]),
            tuple(tuple(row) for row in parameters["correlation"]),
            int(parameters["matrix_sampling_attempt"]),
            str(parameters["dependence_strength"]),
            float(parameters["strength_alpha"]),
        )


@dataclass(frozen=True)
class LowRankGaussianCovariates:
    schema: CovariateSchema
    root_specs: tuple[RootSpec, ...]
    rank: int
    loadings: tuple[tuple[float, ...], ...]
    residual: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    dependence_strength: str
    strength_alpha: float

    @property
    def dimension(self) -> int:
        return self.schema.dimension

    def __post_init__(self) -> None:
        roots = _dependent_schema_roots(self.schema, self.root_specs)
        maximum_rank = min(10, self.schema.dimension - 1)
        if type(self.rank) is not int or not 1 <= self.rank <= maximum_rank:
            raise ValueError("Low-rank Gaussian rank is outside its frozen range.")
        loadings = np.asarray(self.loadings, dtype=np.float64)
        if loadings.shape != (self.schema.dimension, self.rank) or not np.isfinite(
            loadings
        ).all():
            raise ValueError("Low-rank loadings have the wrong shape or values.")
        residual = np.asarray(self.residual, dtype=np.float64)
        if residual.shape != (self.schema.dimension,) or not np.array_equal(
            residual, np.ones(self.schema.dimension)
        ):
            raise ValueError("Low-rank diagonal residual must be frozen at one.")
        correlation = _validate_correlation(self.correlation, self.schema.dimension)
        if (self.dependence_strength, self.strength_alpha) not in _DEPENDENCE_STRENGTHS:
            raise ValueError("Dependence strength label and alpha disagree.")
        expected = _strengthened_correlation(
            _normalized_correlation(loadings @ loadings.T + np.diag(residual)),
            self.strength_alpha,
        )
        if not np.allclose(correlation, expected, rtol=0.0, atol=1e-12):
            raise ValueError("Frozen correlation disagrees with low-rank parameters.")
        object.__setattr__(self, "root_specs", roots)
        object.__setattr__(
            self, "loadings", tuple(tuple(float(x) for x in row) for row in loadings)
        )
        object.__setattr__(self, "residual", tuple(float(x) for x in residual))
        object.__setattr__(
            self, "correlation", tuple(tuple(float(x) for x in row) for row in correlation)
        )

    def sample(self, n: int, rng: np.random.Generator) -> TypedCovariateBatch:
        return _dependent_sample(
            self.schema, self.root_specs, np.asarray(self.correlation), n, rng
        )

    def contains(self, values: Any) -> bool:
        return _dependent_contains(self.schema, self.root_specs, values)


class LowRankGaussianCovariateFactory:
    role = ComponentRole.COVARIATE
    family_id = "bdpfn.covariate.low_rank_gaussian"
    version = "1.0.0"
    semantic_class_id = "covariate.low_rank_gaussian"
    outcome_type = None

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Covariate specification sampling requires a local Generator.")
        schema, roots = _schema_roots_from_hyperparameters(hyperparameters, rng)
        strength, alpha = _sample_dependence_strength(rng)
        rank = int(rng.integers(1, min(10, schema.dimension - 1) + 1))
        loadings = rng.normal(
            scale=1.0 / np.sqrt(rank),
            size=(schema.dimension, rank),
        )
        residual = np.ones(schema.dimension)
        correlation = _strengthened_correlation(
            _normalized_correlation(loadings @ loadings.T + np.diag(residual)),
            alpha,
        )
        _validate_correlation(correlation, schema.dimension)
        return FamilyRef.create(
            self.role.value,
            self.family_id,
            self.version,
            self.semantic_class_id,
            {
                "schema": schema.to_dict(),
                "root_specs": [root.to_dict() for root in roots],
                "rank": rank,
                "loadings": loadings.tolist(),
                "residual": residual.tolist(),
                "correlation": correlation.tolist(),
                "dependence_strength": strength,
                "strength_alpha": alpha,
            },
        )

    def build(self, spec: FamilyRef) -> LowRankGaussianCovariates:
        if (
            not isinstance(spec, FamilyRef)
            or spec.role != self.role.value
            or spec.family_id != self.family_id
            or spec.version != self.version
            or spec.semantic_class_id != self.semantic_class_id
        ):
            raise ValueError("FamilyRef does not describe a low-rank Gaussian law.")
        parameters = spec.parameter_dict()
        expected = {
            "schema", "root_specs", "rank", "loadings", "residual", "correlation",
            "dependence_strength", "strength_alpha"
        }
        if set(parameters) != expected:
            raise ValueError("Low-rank Gaussian specification has unexpected fields.")
        return LowRankGaussianCovariates(
            CovariateSchema.from_dict(parameters["schema"]),
            tuple(RootSpec.from_dict(item) for item in parameters["root_specs"]),
            int(parameters["rank"]),
            tuple(tuple(row) for row in parameters["loadings"]),
            tuple(float(value) for value in parameters["residual"]),
            tuple(tuple(row) for row in parameters["correlation"]),
            str(parameters["dependence_strength"]),
            float(parameters["strength_alpha"]),
        )


@dataclass(frozen=True)
class StudentTCopulaCovariates:
    schema: CovariateSchema
    root_specs: tuple[RootSpec, ...]
    df: float
    gram_factor: tuple[tuple[float, ...], ...]
    correlation: tuple[tuple[float, ...], ...]
    matrix_sampling_attempt: int
    dependence_strength: str
    strength_alpha: float

    @property
    def dimension(self) -> int:
        return self.schema.dimension

    def __post_init__(self) -> None:
        roots = _dependent_schema_roots(self.schema, self.root_specs)
        if not np.isfinite(self.df) or not 3.0 <= self.df <= 32.0:
            raise ValueError("Student-t copula df must lie in [3, 32].")
        if type(self.matrix_sampling_attempt) is not int or self.matrix_sampling_attempt < 0:
            raise ValueError("Matrix sampling attempt must be non-negative.")
        if (self.dependence_strength, self.strength_alpha) not in _DEPENDENCE_STRENGTHS:
            raise ValueError("Dependence strength label and alpha disagree.")
        factor = _matrix(self.gram_factor, self.schema.dimension, "Gram factor")
        correlation = _validate_correlation(self.correlation, self.schema.dimension)
        expected = _strengthened_correlation(
            _normalized_correlation(factor @ factor.T), self.strength_alpha
        )
        if not np.allclose(correlation, expected, rtol=0.0, atol=1e-12):
            raise ValueError("Student-t correlation disagrees with its frozen parameters.")
        object.__setattr__(self, "root_specs", roots)
        object.__setattr__(self, "df", float(self.df))
        object.__setattr__(
            self, "gram_factor", tuple(tuple(float(x) for x in row) for row in factor)
        )
        object.__setattr__(
            self, "correlation", tuple(tuple(float(x) for x in row) for row in correlation)
        )

    def sample(self, n: int, rng: np.random.Generator) -> TypedCovariateBatch:
        return _dependent_sample(
            self.schema,
            self.root_specs,
            np.asarray(self.correlation),
            n,
            rng,
            student_df=self.df,
        )

    def contains(self, values: Any) -> bool:
        return _dependent_contains(self.schema, self.root_specs, values)

    def upper_tail_dependence(self, left: int, right: int) -> float:
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left < self.schema.dimension
            or not 0 <= right < self.schema.dimension
        ):
            raise ValueError("Tail-dependence indices lie outside the schema.")
        if left == right:
            return 1.0
        rho = self.correlation[left][right]
        argument = -np.sqrt((self.df + 1.0) * (1.0 - rho) / (1.0 + rho))
        return float(2.0 * stdtr(self.df + 1.0, argument))


class StudentTCopulaCovariateFactory:
    role = ComponentRole.COVARIATE
    family_id = "bdpfn.covariate.student_t_copula"
    version = "1.0.0"
    semantic_class_id = "covariate.student_t_copula"
    outcome_type = None

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Covariate specification sampling requires a local Generator.")
        schema, roots = _schema_roots_from_hyperparameters(hyperparameters, rng)
        df = 2.0 + float(np.exp(rng.uniform(np.log(1.0), np.log(30.0))))
        strength, alpha = _sample_dependence_strength(rng)
        max_attempts = int(hyperparameters.get("max_matrix_attempts", 100))
        if max_attempts <= 0:
            raise ValueError("Matrix resampling limit must be positive.")
        for attempt in range(max_attempts):
            factor = rng.normal(size=(schema.dimension, schema.dimension))
            correlation = _strengthened_correlation(
                _normalized_correlation(factor @ factor.T), alpha
            )
            try:
                _validate_correlation(correlation, schema.dimension)
            except ValueError:
                continue
            return FamilyRef.create(
                self.role.value,
                self.family_id,
                self.version,
                self.semantic_class_id,
                {
                    "schema": schema.to_dict(),
                    "root_specs": [root.to_dict() for root in roots],
                    "df": df,
                    "gram_factor": factor.tolist(),
                    "correlation": correlation.tolist(),
                    "matrix_sampling_attempt": attempt,
                    "dependence_strength": strength,
                    "strength_alpha": alpha,
                },
            )
        raise ValueError("Student-t Gram sampling exhausted its retry limit.")

    def build(self, spec: FamilyRef) -> StudentTCopulaCovariates:
        if (
            not isinstance(spec, FamilyRef)
            or spec.role != self.role.value
            or spec.family_id != self.family_id
            or spec.version != self.version
            or spec.semantic_class_id != self.semantic_class_id
        ):
            raise ValueError("FamilyRef does not describe a Student-t copula.")
        parameters = spec.parameter_dict()
        expected = {
            "schema", "root_specs", "df", "gram_factor", "correlation",
            "matrix_sampling_attempt", "dependence_strength", "strength_alpha"
        }
        if set(parameters) != expected:
            raise ValueError("Student-t copula specification has unexpected fields.")
        return StudentTCopulaCovariates(
            CovariateSchema.from_dict(parameters["schema"]),
            tuple(RootSpec.from_dict(item) for item in parameters["root_specs"]),
            float(parameters["df"]),
            tuple(tuple(row) for row in parameters["gram_factor"]),
            tuple(tuple(row) for row in parameters["correlation"]),
            int(parameters["matrix_sampling_attempt"]),
            str(parameters["dependence_strength"]),
            float(parameters["strength_alpha"]),
        )


@dataclass(frozen=True)
class IndependentStandardNormalCovariates:
    dimension: int

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Covariate dimension must be positive.")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("Sample size must be non-negative.")
        return rng.normal(size=(n, self.dimension))

    def contains(self, x: np.ndarray) -> bool:
        return (
            isinstance(x, np.ndarray)
            and x.ndim == 2
            and x.shape[1] == self.dimension
            and bool(np.isfinite(x).all())
        )


class IndependentStandardNormalFactory:
    role = ComponentRole.COVARIATE
    family_id = "bdpfn.covariate.independent_standard_normal"
    version = "1.0.0"
    semantic_class_id = "covariate.independent_standard_normal"
    outcome_type = None

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef:
        dimension = int(hyperparameters["dimension"])
        return FamilyRef.create(
            role=self.role.value,
            family_id=self.family_id,
            version=self.version,
            semantic_class_id=self.semantic_class_id,
            parameters={"dimension": dimension},
        )

    def build(self, spec: FamilyRef) -> IndependentStandardNormalCovariates:
        return IndependentStandardNormalCovariates(
            int(spec.parameter_dict()["dimension"])
        )


@dataclass(frozen=True)
class IndependentProductCovariates:
    schema: CovariateSchema
    root_specs: tuple[RootSpec, ...]

    @property
    def dimension(self) -> int:
        return self.schema.dimension

    def __post_init__(self) -> None:
        if not isinstance(self.schema, CovariateSchema):
            raise TypeError("Independent-product law requires a CovariateSchema.")
        roots = tuple(self.root_specs)
        object.__setattr__(self, "root_specs", roots)
        if len(roots) != self.schema.dimension:
            raise ValueError("Root count disagrees with the covariate schema.")
        for variable, root in zip(self.schema.variables, roots):
            if (
                root.variable_type is not variable.variable_type
                or root.cardinality != variable.cardinality
            ):
                raise ValueError("Root specification disagrees with its variable.")

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> TypedCovariateBatch:
        if type(n) is not int or n < 0:
            raise ValueError("Covariate sample size must be non-negative.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Covariate sampling requires a local NumPy Generator.")
        return TypedCovariateBatch(
            self.schema,
            tuple(build_root(root).sample(n, rng) for root in self.root_specs),
        )

    def contains(self, values: Any) -> bool:
        try:
            if isinstance(values, TypedCovariateBatch):
                if values.schema != self.schema:
                    return False
                batch = values
            else:
                batch = TypedCovariateBatch.from_matrix(
                    self.schema,
                    np.asarray(values),
                )
        except (TypeError, ValueError):
            return False
        return all(
            build_root(root).contains(column)
            for root, column in zip(self.root_specs, batch.columns)
        )


class IndependentProductCovariateFactory:
    role = ComponentRole.COVARIATE
    family_id = "bdpfn.covariate.independent_product"
    version = "1.0.0"
    semantic_class_id = "covariate.independent_product"
    outcome_type = None

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Covariate specification sampling requires a local Generator.")
        if "schema" in hyperparameters:
            schema = CovariateSchema.from_dict(hyperparameters["schema"])
        else:
            schema = sample_schema(rng)
        root_specs = tuple(sample_root_spec(variable, rng) for variable in schema.variables)
        return FamilyRef.create(
            role=self.role.value,
            family_id=self.family_id,
            version=self.version,
            semantic_class_id=self.semantic_class_id,
            parameters={
                "schema": schema.to_dict(),
                "root_specs": [root.to_dict() for root in root_specs],
            },
        )

    def build(self, spec: FamilyRef) -> IndependentProductCovariates:
        if not isinstance(spec, FamilyRef):
            raise TypeError("Covariate build input must be a FamilyRef.")
        if (
            spec.role != self.role.value
            or spec.family_id != self.family_id
            or spec.version != self.version
            or spec.semantic_class_id != self.semantic_class_id
        ):
            raise ValueError("FamilyRef does not describe this covariate factory.")
        parameters = spec.parameter_dict()
        if set(parameters) != {"schema", "root_specs"}:
            raise ValueError("Independent-product specification has unexpected fields.")
        return IndependentProductCovariates(
            schema=CovariateSchema.from_dict(parameters["schema"]),
            root_specs=tuple(
                RootSpec.from_dict(item) for item in parameters["root_specs"]
            ),
        )


@dataclass(frozen=True)
class EmpiricalRowBootstrapCovariates:
    artifact: ResolvedArtifact
    sampling_mode: str

    def __post_init__(self) -> None:
        if self.sampling_mode not in {"uniform", "weighted"}:
            raise ValueError("Unknown empirical row sampling mode.")
        if self.sampling_mode == "weighted" and self.artifact.normalized_weights is None:
            raise ValueError("Weighted sampling requires frozen artifact weights.")

    @property
    def dimension(self) -> int:
        return self.artifact.schema.dimension

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> TypedCovariateBatch:
        if type(n) is not int or n < 0:
            raise ValueError("Covariate sample size must be non-negative.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Covariate sampling requires a local NumPy Generator.")
        uniforms = rng.random(n)
        if self.sampling_mode == "uniform":
            indices = np.floor(self.artifact.batch.n_rows * uniforms).astype(np.int64)
        else:
            cumulative = np.cumsum(
                np.asarray(self.artifact.normalized_weights, dtype=np.float64)
            )
            cumulative[-1] = 1.0
            indices = np.searchsorted(cumulative, uniforms, side="right")
        return TypedCovariateBatch(
            self.artifact.schema,
            tuple(column[indices] for column in self.artifact.batch.columns),
        )

    def contains(self, values: Any) -> bool:
        try:
            batch = (
                values
                if isinstance(values, TypedCovariateBatch)
                else TypedCovariateBatch.from_matrix(
                    self.artifact.schema, np.asarray(values)
                )
            )
        except (TypeError, ValueError):
            return False
        if batch.schema != self.artifact.schema:
            return False
        table = self.artifact.batch.to_matrix()
        query = batch.to_matrix()
        return bool(
            np.all(
                np.any(np.all(query[:, None, :] == table[None, :, :], axis=2), axis=1)
            )
        )


class EmpiricalRowBootstrapFactory:
    role = ComponentRole.COVARIATE
    family_id = "bdpfn.covariate.empirical_row_bootstrap"
    version = "1.0.0"
    semantic_class_id = "covariate.empirical_row_bootstrap"
    outcome_type = None

    def __init__(self, artifact_resolver: ArtifactResolver | None = None) -> None:
        self._artifact_resolver = artifact_resolver

    def _resolve(self, artifact_id: str, artifact_sha256: str) -> ResolvedArtifact:
        if self._artifact_resolver is None:
            raise ValueError(
                "Empirical covariates require an explicitly injected artifact resolver."
            )
        return self._artifact_resolver.resolve(artifact_id, artifact_sha256)

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Covariate specification sampling requires a local Generator.")
        expected = {"artifact_id", "artifact_sha256", "sampling_mode"}
        if set(hyperparameters) != expected:
            raise ValueError("Empirical-covariate hyperparameters have unexpected fields.")
        artifact_id = str(hyperparameters["artifact_id"])
        artifact_sha256 = str(hyperparameters["artifact_sha256"])
        sampling_mode = str(hyperparameters["sampling_mode"])
        if sampling_mode not in {"uniform", "weighted"}:
            raise ValueError("Unknown empirical row sampling mode.")
        artifact = self._resolve(artifact_id, artifact_sha256)
        if sampling_mode == "weighted" and artifact.normalized_weights is None:
            raise ValueError("Weighted sampling requires frozen artifact weights.")
        return FamilyRef.create(
            self.role.value,
            self.family_id,
            self.version,
            self.semantic_class_id,
            {
                "artifact_id": artifact.artifact_id,
                "artifact_sha256": artifact.artifact_sha256,
                "table_sha256": artifact.table_sha256,
                "weights_sha256": artifact.weights_sha256,
                "provenance_sha256": artifact.provenance_sha256,
                "schema": artifact.schema.to_dict(),
                "sampling_mode": sampling_mode,
            },
        )

    def build(self, spec: FamilyRef) -> EmpiricalRowBootstrapCovariates:
        if (
            not isinstance(spec, FamilyRef)
            or spec.role != self.role.value
            or spec.family_id != self.family_id
            or spec.version != self.version
            or spec.semantic_class_id != self.semantic_class_id
        ):
            raise ValueError("FamilyRef does not describe empirical row bootstrap.")
        parameters = spec.parameter_dict()
        expected = {
            "artifact_id", "artifact_sha256", "table_sha256", "weights_sha256",
            "provenance_sha256", "schema", "sampling_mode"
        }
        if set(parameters) != expected:
            raise ValueError("Empirical-covariate specification has unexpected fields.")
        artifact = self._resolve(
            str(parameters["artifact_id"]), str(parameters["artifact_sha256"])
        )
        declared = {
            "table_sha256": artifact.table_sha256,
            "weights_sha256": artifact.weights_sha256,
            "provenance_sha256": artifact.provenance_sha256,
            "schema": artifact.schema.to_dict(),
        }
        if any(parameters[key] != value for key, value in declared.items()):
            raise ValueError("Resolved artifact metadata disagrees with its frozen spec.")
        return EmpiricalRowBootstrapCovariates(
            artifact, str(parameters["sampling_mode"])
        )


def covariate_dependence_law(
    schema: CovariateSchema,
) -> tuple[tuple[str, float], ...]:
    """Return the frozen schema-conditional X-coordinate dependence law."""

    if not isinstance(schema, CovariateSchema):
        raise TypeError("Dependence-law audit requires a CovariateSchema.")
    if schema.dimension >= 2:
        return (
            ("independent_product", 0.10),
            ("gaussian_copula", 0.30),
            ("low_rank_gaussian", 0.20),
            ("student_t_copula", 0.40),
        )
    return (("independent_product", 1.0),)


def sample_covariate_spec(
    rng: np.random.Generator,
    schema: CovariateSchema | None = None,
) -> FamilyRef:
    """Sample and freeze one schema plus its permitted dependence family."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("Covariate chooser requires a local NumPy Generator.")
    if schema is None:
        frozen_schema = sample_schema(rng)
    elif isinstance(schema, CovariateSchema):
        frozen_schema = schema
    else:
        raise TypeError("Covariate chooser schema must be a CovariateSchema.")
    law = covariate_dependence_law(frozen_schema)
    probabilities = np.asarray([weight for _, weight in law], dtype=np.float64)
    family = law[int(rng.choice(len(law), p=probabilities))][0]
    factories = {
        "independent_product": IndependentProductCovariateFactory(),
        "gaussian_copula": GaussianCopulaCovariateFactory(),
        "low_rank_gaussian": LowRankGaussianCovariateFactory(),
        "student_t_copula": StudentTCopulaCovariateFactory(),
    }
    return factories[family].sample_spec(
        rng,
        {"schema": frozen_schema.to_dict()},
        {},
    )


def build_covariates(spec: FamilyRef) -> Any:
    """Build one covariate law supported by the schema-conditional chooser."""

    if not isinstance(spec, FamilyRef):
        raise TypeError("Covariate dispatch requires a FamilyRef.")
    factories = {
        IndependentProductCovariateFactory.family_id: IndependentProductCovariateFactory(),
        GaussianCopulaCovariateFactory.family_id: GaussianCopulaCovariateFactory(),
        LowRankGaussianCovariateFactory.family_id: LowRankGaussianCovariateFactory(),
        StudentTCopulaCovariateFactory.family_id: StudentTCopulaCovariateFactory(),
    }
    try:
        factory = factories[spec.family_id]
    except KeyError as error:
        raise ValueError("Unsupported covariate family for chooser dispatch.") from error
    return factory.build(spec)
