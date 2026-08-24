"""Frozen active-v1 design manifest; no active-v1 sampler is enabled yet."""

from __future__ import annotations

from typing import Any, Mapping

from .manifest import (
    ActivePriorManifest,
    DesignChoice,
    DesignLaw,
    FamilyDesign,
)
from .specs import canonical_hash


def _law(
    law_id: str,
    choices: Mapping[str, float],
    metadata: Mapping[str, Any] | None = None,
) -> DesignLaw:
    return DesignLaw.create(
        law_id,
        tuple(
            DesignChoice.create(label, weight)
            for label, weight in choices.items()
        ),
        metadata,
    )


def _family(
    role: str,
    family_id: str,
    partition: str,
    conditional_weight: float | None = None,
    weight_scope: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FamilyDesign:
    return FamilyDesign.create(
        role=role,
        family_id=family_id,
        version="1.0.0",
        partition=partition,
        conditional_weight=conditional_weight,
        weight_scope=weight_scope,
        metadata=metadata,
    )


def _families() -> tuple[FamilyDesign, ...]:
    return (
        _family("root", "bdpfn.root.normal", "train", 0.20, "root.continuous"),
        _family("root", "bdpfn.root.uniform", "train", 0.15, "root.continuous"),
        _family(
            "root",
            "bdpfn.root.beta_standardized",
            "train",
            0.10,
            "root.continuous",
        ),
        _family(
            "root",
            "bdpfn.root.laplace_standardized",
            "train",
            0.15,
            "root.continuous",
        ),
        _family(
            "root",
            "bdpfn.root.gaussian_mixture2_standardized",
            "train",
            0.15,
            "root.continuous",
        ),
        _family(
            "root",
            "bdpfn.root.student_t_standardized",
            "train",
            0.25,
            "root.continuous",
            {"df_range": [3.0, 32.0], "fourth_moment_absent_when_df_le_4": True},
        ),
        _family("root", "bdpfn.root.bernoulli", "train", 1.0, "root.binary"),
        _family(
            "root",
            "bdpfn.root.categorical_uniform",
            "train",
            0.35,
            "root.categorical_pmf",
        ),
        _family(
            "root",
            "bdpfn.root.categorical_dirichlet1",
            "train",
            0.35,
            "root.categorical_pmf",
        ),
        _family(
            "root",
            "bdpfn.root.categorical_truncated_zipf",
            "train",
            0.30,
            "root.categorical_pmf",
        ),
        _family(
            "root",
            "bdpfn.root.ordinal_randint",
            "train",
            0.50,
            "root.ordinal",
            {"selection_rule": "uniform_over_compatible_pool"},
        ),
        _family(
            "root",
            "bdpfn.root.ordinal_truncated_zipf",
            "train",
            0.50,
            "root.ordinal",
            {"selection_rule": "uniform_over_compatible_pool"},
        ),
        _family(
            "covariate",
            "bdpfn.covariate.independent_product",
            "train",
            0.10,
            "covariate.coordinate_dependence",
        ),
        _family(
            "covariate",
            "bdpfn.covariate.gaussian_copula",
            "train",
            0.30,
            "covariate.coordinate_dependence",
        ),
        _family(
            "covariate",
            "bdpfn.covariate.low_rank_gaussian",
            "train",
            0.20,
            "covariate.coordinate_dependence",
        ),
        _family(
            "covariate",
            "bdpfn.covariate.student_t_copula",
            "train",
            0.40,
            "covariate.coordinate_dependence",
            {"latent_tail_dependence": "nonzero", "df_range": [3.0, 32.0]},
        ),
        _family(
            "propensity",
            "bdpfn.propensity.constant_rct",
            "train",
            0.20,
            "propensity.family",
        ),
        _family(
            "propensity",
            "bdpfn.propensity.sparse_affine",
            "train",
            0.35,
            "propensity.family",
        ),
        _family(
            "propensity",
            "bdpfn.propensity.dense_affine",
            "train",
            0.15,
            "propensity.family",
        ),
        _family(
            "propensity",
            "bdpfn.propensity.rff",
            "train",
            0.20,
            "propensity.family",
        ),
        _family(
            "propensity",
            "bdpfn.propensity.small_mlp",
            "train",
            0.10,
            "propensity.family",
        ),
        *(
            _family("scalar", f"bdpfn.scalar.{family_id}", "train", weight, "scalar.family")
            for family_id, weight in (
                ("constant", 0.05),
                ("projection", 0.05),
                ("sparse_affine", 0.15),
                ("dense_affine", 0.10),
                ("categorical_lookup", 0.08),
                ("threshold_or_stump", 0.08),
                ("shallow_tree_or_forest", 0.08),
                ("polynomial", 0.10),
                ("gam_hinge_or_spline", 0.10),
                ("rbf", 0.08),
                ("rff_or_fourier", 0.08),
                ("small_mlp", 0.05),
            )
        ),
        _family(
            "outcome",
            "bdpfn.outcome.continuous_centered_gaussian",
            "train",
            1.0 / 3.0,
            "outcome.likelihood.continuous",
        ),
        _family(
            "outcome",
            "bdpfn.outcome.continuous_centered_laplace",
            "train",
            1.0 / 3.0,
            "outcome.likelihood.continuous",
        ),
        _family(
            "outcome",
            "bdpfn.outcome.continuous_bounded_uniform",
            "train",
            1.0 / 3.0,
            "outcome.likelihood.continuous",
        ),
        _family(
            "outcome",
            "bdpfn.outcome.bounded_beta",
            "train",
            0.50,
            "outcome.likelihood.bounded_continuous",
            {"selection_rule": "uniform_over_compatible_pool"},
        ),
        _family(
            "outcome",
            "bdpfn.outcome.bounded_uniform",
            "train",
            0.50,
            "outcome.likelihood.bounded_continuous",
            {"selection_rule": "uniform_over_compatible_pool"},
        ),
        _family(
            "outcome",
            "bdpfn.outcome.binary_bernoulli",
            "train",
            1.0,
            "outcome.likelihood.binary",
        ),
        _family(
            "outcome",
            "bdpfn.outcome.count_binomial",
            "train",
            0.50,
            "outcome.likelihood.count",
            {"selection_rule": "uniform_over_compatible_pool"},
        ),
        _family(
            "outcome",
            "bdpfn.outcome.count_poisson",
            "train",
            0.50,
            "outcome.likelihood.count",
            {"selection_rule": "uniform_over_compatible_pool"},
        ),
        _family(
            "outcome",
            "bdpfn.outcome.numeric_categorical",
            "train",
            1.0,
            "outcome.likelihood.numeric_categorical_or_ordinal",
        ),
        _family(
            "outcome",
            "bdpfn.outcome.nominal_categorical",
            "train",
            1.0,
            "outcome.likelihood.nominal_categorical",
        ),
        _family("outcome", "bdpfn.outcome.negative_binomial", "optional"),
        _family(
            "root",
            "bdpfn.root.empirical_bgm",
            "optional",
            metadata={"activation": "plugin_only"},
        ),
        _family(
            "scalar",
            "bdpfn.scalar.unrestricted_callable",
            "optional",
            metadata={"activation": "plugin_only"},
        ),
        _family("scalar", "bdpfn.scalar.log", "held_out"),
        _family("scalar", "bdpfn.scalar.exp", "held_out"),
        _family("scalar", "bdpfn.scalar.gp_finite_point", "held_out"),
        _family("root", "bdpfn.root.cauchy", "excluded"),
        _family("outcome", "bdpfn.outcome.cauchy", "excluded"),
        _family(
            "causal_mode",
            "bdpfn.causal_mode.direct_treatment_outcome_hidden_confounding",
            "excluded",
        ),
    )


def _laws() -> tuple[DesignLaw, ...]:
    uniform_eighths = {str(value): 1.0 / 8.0 for value in range(3, 11)}
    return (
        _law(
            "dimension.stratum",
            {
                "1": 1.0 / 6.0,
                "2-5": 1.0 / 6.0,
                "6-10": 1.0 / 6.0,
                "11-20": 1.0 / 6.0,
                "21-50": 1.0 / 6.0,
                "51-99": 1.0 / 6.0,
            },
            {"within_non_singleton_stratum": "discrete_uniform"},
        ),
        _law(
            "context.size",
            {str(value): 1.0 / 9.0 for value in (8, 16, 32, 64, 128, 256, 512, 1024, 2048)},
        ),
        _law(
            "query.size",
            {str(value): 0.25 for value in (64, 128, 256, 512)},
            {"query_rows": "iid_from_P_X"},
        ),
        _law(
            "schema.profile",
            {
                "continuous_only": 0.30,
                "binary_only": 0.10,
                "categorical_or_ordinal_only": 0.10,
                "mixed": 0.50,
            },
            {"incompatible_dimension_policy": "later_rejection_sampling"},
        ),
        _law(
            "schema.mixed_coordinate_type",
            {
                "continuous": 0.50,
                "binary": 0.20,
                "categorical": 0.15,
                "ordinal_or_integer": 0.15,
            },
        ),
        _law(
            "outcome.type",
            {
                "continuous": 0.40,
                "bounded_continuous": 0.10,
                "binary": 0.20,
                "count": 0.10,
                "numeric_categorical_or_ordinal": 0.10,
                "nominal_categorical": 0.10,
            },
        ),
        _law("arm.likelihood_coupling", {"same": 0.70, "different_compatible": 0.30}),
        _law("arm.mean_family_coupling", {"same": 0.60, "different": 0.40}),
        _law("arm.same_family_backbone", {"shared": 0.50, "independent": 0.50}),
        _law(
            "covariate.categorical_cardinality_band",
            {"2": 0.40, "3-9": 0.40, "10-30": 0.20},
            {"within_band": "probability_proportional_to_inverse_k"},
        ),
        _law("outcome.class_count", uniform_eighths, {"selection": "discrete_uniform"}),
        _law(
            "root.continuous",
            {"normal": 0.20, "uniform": 0.15, "beta": 0.10, "laplace": 0.15, "gaussian_mixture2": 0.15, "student_t": 0.25},
            {"student_t_df": "2+exp(Uniform(log(1),log(30)))", "fourth_moment_absent_when_df_le_4": True},
        ),
        _law(
            "root.categorical_pmf",
            {"uniform": 0.35, "dirichlet1": 0.35, "truncated_zipf": 0.30},
            {"cell_floor_mixture": "0.9*p+0.1/K"},
        ),
        _law("root.binary", {"bernoulli": 1.0}),
        _law(
            "root.ordinal",
            {"randint": 0.50, "truncated_zipf": 0.50},
            {"selection_rule": "uniform_over_compatible_pool"},
        ),
        _law("root.binary_probability", {"point_mass_0.5": 0.40, "uniform_0.1_0.9": 0.60}),
        _law(
            "covariate.coordinate_dependence",
            {"independent": 0.10, "gaussian_copula": 0.30, "low_rank_gaussian": 0.20, "student_t_copula": 0.40},
            {"d_equals_1": "independent_product", "mixed_discrete": "serialized_latent_generalized_inverse_cdf"},
        ),
        _law("propensity.epsilon", {"0.10": 0.50, "0.05": 0.35, "0.02": 0.15}),
        _law(
            "propensity.family",
            {"constant_rct": 0.20, "sparse_affine": 0.35, "dense_affine": 0.15, "rff": 0.20, "small_mlp": 0.10},
        ),
        _law("propensity.logit_amplitude", {"0.5": 0.25, "1": 0.50, "2": 0.25}),
        _law(
            "scalar.family",
            {
                "constant": 0.05,
                "projection": 0.05,
                "sparse_affine": 0.15,
                "dense_affine": 0.10,
                "categorical_lookup": 0.08,
                "threshold_or_stump": 0.08,
                "shallow_tree_or_forest": 0.08,
                "polynomial": 0.10,
                "gam_hinge_or_spline": 0.10,
                "rbf": 0.08,
                "rff_or_fourier": 0.08,
                "small_mlp": 0.05,
            },
        ),
        _law("scalar.function_amplitude", {"0.5": 0.25, "1": 0.50, "2": 0.25}),
        _law("scalar.treatment_effect_amplitude", {"0.25": 0.20, "0.5": 0.40, "1": 0.30, "2": 0.10}),
        _law("scalar.polynomial_degree", {"1": 0.35, "2": 0.30, "3": 0.20, "4": 0.15}),
        _law("scalar.rff_feature_count", {"32": 0.25, "64": 0.50, "128": 0.25}),
        _law("scalar.mlp_width", {"16": 0.25, "32": 0.25, "64": 0.25, "128": 0.25}),
        _law("scalar.tree_count", {"1": 0.25, "4": 0.25, "8": 0.25, "16": 0.25}),
        _law("composition.mode", {"atomic_only": 0.60, "composed": 0.40}),
        _law("outcome.continuous_noise_mode", {"deterministic": 0.20, "stochastic": 0.80}),
        _law("outcome.continuous_residual", {"gaussian": 1.0 / 3.0, "laplace": 1.0 / 3.0, "bounded_uniform": 1.0 / 3.0}),
        _law("outcome.continuous_heteroscedastic", {"homoscedastic": 0.70, "heteroscedastic": 0.30}),
        _law("outcome.likelihood.continuous", {"centered_gaussian": 1.0 / 3.0, "centered_laplace": 1.0 / 3.0, "bounded_uniform": 1.0 / 3.0}, {"different_coupling_policy": "sample_distinct_when_available"}),
        _law("outcome.likelihood.bounded_continuous", {"beta": 0.50, "bounded_uniform": 0.50}, {"selection_rule": "uniform_over_compatible_pool", "different_coupling_policy": "sample_distinct_when_available"}),
        _law("outcome.likelihood.binary", {"bernoulli": 1.0}, {"different_coupling_policy": "same_only_if_singleton"}),
        _law("outcome.likelihood.count", {"binomial": 0.50, "poisson": 0.50}, {"selection_rule": "uniform_over_compatible_pool", "different_coupling_policy": "sample_distinct_when_available"}),
        _law("outcome.likelihood.numeric_categorical_or_ordinal", {"finite_numeric_categorical": 1.0}, {"different_coupling_policy": "same_only_if_singleton"}),
        _law("outcome.likelihood.nominal_categorical", {"nominal_categorical": 1.0}, {"different_coupling_policy": "same_only_if_singleton"}),
    )


def _caps() -> dict[str, Any]:
    return {
        "covariate_dimension_max": 99,
        "categorical_cardinality": {"train_max": 30, "held_out": [31, 100], "plugin_hard_cap": 256},
        "outcome_class_count": {"min": 3, "max": 10},
        "root_parameters": {
            "normal": {"mean": 0.0, "sd": 1.0},
            "uniform": {"low": "-sqrt(3)", "high": "sqrt(3)"},
            "laplace": {"location": 0.0, "scale": "1/sqrt(2)"},
            "beta": {"alpha_loguniform": [0.5, 5.0], "beta_loguniform": [0.5, 5.0], "standardization": "exact_marginal"},
            "gaussian_mixture2": {"weight_uniform": [0.2, 0.8], "mean_separation_uniform": [0.5, 3.0], "component_sd_loguniform": [0.2, 1.5], "standardization": "exact_marginal"},
            "truncated_zipf_exponent_uniform": [0.5, 2.0],
        },
        "coefficient_law": "N(0,1/k)",
        "intercept_law": "N(0,1)",
        "scalar_caps": {
            "sparse_active_variables_train": "1..min(10,d_X)",
            "sparse_active_variables_held_out": "11..30_if_compatible",
            "pair_interactions_train": [1, 9],
            "pair_interactions_held_out": [10, 30],
            "polynomial_degree_held_out": [5, 8],
            "rff_features_held_out": [256],
            "rff_length_scale_train_loguniform": [0.5, 3.0],
            "rff_length_scale_held_out": [[0.1, 0.5], [3.0, 10.0]],
            "mlp_depth_train": [2, 4],
            "mlp_depth_held_out": [5, 6],
            "mlp_width_held_out": [256, 512],
            "tree_count_held_out": [32, 128],
            "tree_depth_train": [1, 5],
            "tree_depth_held_out": [6, 7],
            "spline_knots_train": [1, 6],
            "spline_knots_held_out": [7, 16],
            "composition_depth_train_max": 3,
            "composition_atomic_nodes_train_max": 7,
            "composition_depth_held_out": [4, 5],
        },
        "continuous_noise": {
            "signal_variance_floor": 1e-4,
            "noise_to_signal_ratio_uniform": [0.05, 1.0],
            "relative_scale_clip": [0.2, 2.0],
            "calibration_gate": "unadmitted_until_exact_or_certified",
        },
        "certified_numeric": {"atol_max": 1e-8, "rtol_max": 1e-6},
        "compatibility": {
            "singleton_likelihood_pool": "same_only_if_singleton",
            "incompatible_schema_dimension": "later_rejection_sampling",
            "nonlinear_calibration": "unadmitted_until_exact_or_certified",
        },
    }


def active_v1_manifest() -> ActivePriorManifest:
    families = _families()
    registry_entries = [
        {
            "role": family.role,
            "family_id": family.family_id,
            "version": family.version,
            "partition": family.partition,
        }
        for family in sorted(
            families,
            key=lambda item: (item.role, item.family_id, item.version),
        )
    ]
    required = tuple(
        sorted(
            family.family_id
            for family in families
            if family.partition == "train"
        )
    )
    return ActivePriorManifest.create(
        manifest_version="bdpfn-active-v1-design-v1",
        registry_snapshot_digest=canonical_hash(
            {"registry_entries": registry_entries}
        ),
        families=families,
        laws=_laws(),
        caps=_caps(),
        required_admissions=required,
        admission_certificates=(),
    )


__all__ = ["active_v1_manifest"]
