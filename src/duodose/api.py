"""Public Python API for DuoDose."""

from __future__ import annotations

import logging
from typing import Any

from .config import DLConfig, DuoDoseConfig, SemiRealConfig
from .models.registry import DEFAULT_DUODOSE_BACKEND, get_backend_spec
from .postprocess import probability_scores_frame
from .rf_weighting import FORMAL_HIGH_RNA_NEGATIVE_WEIGHT, formal_rf_sample_weights
from .result import DuoDoseResult
from .thresholds import resolve_threshold
from .validation import counts_copy, validate_adata


LOGGER = logging.getLogger(__name__)


class DuoDose:
    """Homotypic-aware doublet detector using a retained supervised backend."""

    def __init__(
        self,
        backend: str = DEFAULT_DUODOSE_BACKEND,
        expected_doublet_rate: float = 0.08,
        random_state: int = 0,
        device: str = "auto",
        *,
        layer: str | None = None,
        threshold_strategy: str | None = "expected_rate",
        threshold: float = 0.5,
        training_preset: str = "default",
        amp: bool = False,
        dl_batch_size: int | None = None,
        dl_max_epochs: int = 200,
        dl_patience: int = 20,
        config: DuoDoseConfig | None = None,
    ) -> None:
        if config is None:
            config = DuoDoseConfig(
                backend=backend,
                expected_doublet_rate=expected_doublet_rate,
                random_state=random_state,
                device=device,
                layer=layer,
                threshold_strategy=threshold_strategy,
                threshold=threshold,
                training_preset=training_preset,
                dl=DLConfig(
                    max_epochs=dl_max_epochs,
                    patience=dl_patience,
                    batch_size=dl_batch_size,
                    amp=amp,
                ),
            )
        self.config = config
        self.backend_ = None
        self.training_summary_: dict[str, Any] = {}
        self.feature_audit_: dict[str, Any] = {}
        self.parent_audit_: dict[str, Any] = {}
        self.construction_report_: dict[str, Any] = {}
        self.safe_feature_transformer_ = None

    def fit(self, adata) -> "DuoDose":
        LOGGER.info("Fitting DuoDose backend=%s on %d cells", self.config.backend, adata.n_obs)
        validate_adata(
            adata,
            layer=self.config.layer,
            expected_doublet_rate=self.config.expected_doublet_rate,
            device=self.config.device,
        )
        work = counts_copy(adata, layer=self.config.layer)
        # Public fitting treats input cells as the unlabeled real-cell background.
        # Existing experimental annotations are never consumed by this path.
        work.obs["experimental_doublet"] = 0

        from .safe_feature_transformer import SafeFeatureTransformer
        from .semireal_bundle import make_parent_disjoint_semireal_bundle
        from .net import train_predict_diagnostic_model

        semireal_config = self.config.semireal or SemiRealConfig.from_preset(self.config.training_preset)
        if work.n_obs < int(semireal_config.minimum_singlets):
            raise ValueError(
                "DuoDose needs at least "
                f"{int(semireal_config.minimum_singlets)} cells for the configured parent-disjoint preset; "
                f"received {work.n_obs}"
            )
        effective_n_singlets = min(int(semireal_config.n_singlets), int(work.n_obs // 2))
        bundle = make_parent_disjoint_semireal_bundle(
            work,
            dataset="duodose_api",
            seed=int(self.config.random_state),
            n_singlets=effective_n_singlets,
            n_train_homotypic_doublets=int(semireal_config.n_train_homotypic_doublets),
            n_train_heterotypic_doublets=int(semireal_config.n_train_heterotypic_doublets),
            n_test_homotypic_doublets=int(semireal_config.n_validation_homotypic_doublets),
            n_test_heterotypic_doublets=int(semireal_config.n_validation_heterotypic_doublets),
            n_clusters=int(semireal_config.n_clusters),
            test_parent_fraction=float(semireal_config.test_parent_fraction),
            validation_parent_fraction=float(semireal_config.validation_parent_fraction),
            high_rna_quantile=float(semireal_config.high_rna_quantile),
            saturation_range=(float(semireal_config.saturation_low), float(semireal_config.saturation_high)),
            min_cluster_size=int(semireal_config.min_cluster_size),
            construction_variant=semireal_config.construction_variant,
        )

        feature_config = self.config.feature
        origin = bundle.fit_adata.obs["semireal_origin"].astype(str)
        reference = bundle.fit_adata[origin.isin({"observed_background", "real_labeled_singlet"}).to_numpy(), :].copy()
        reference_pool_id = (
            "duodose_api|"
            f"seed={int(self.config.random_state)}|"
            f"variant={semireal_config.construction_variant}|fit_split_clean_singlets"
        )
        transformer = SafeFeatureTransformer(
            random_state=int(self.config.random_state),
            reference_seed=int(self.config.random_state),
            n_components=int(feature_config.n_pcs),
            n_clusters=int(semireal_config.n_clusters),
            n_artificial_doublets=feature_config.n_simulated_doublets,
        ).fit(reference, reference_pool_id=reference_pool_id, dataset="duodose_api")
        fit_scores = transformer.transform(
            bundle.fit_adata,
            dataset_id="duodose_api_fit",
            random_state=int(self.config.random_state),
        )
        validation_scores = transformer.transform(
            bundle.val_adata,
            dataset_id="duodose_api_validation",
            random_state=int(self.config.random_state),
        )

        import pandas as pd

        training_scores = pd.concat([fit_scores, validation_scores], axis=0)
        spec = get_backend_spec(self.config.backend)
        train_kwargs: dict[str, Any] = {
            "train_cell_scores": training_scores,
            "test_cell_scores": validation_scores,
            "method": spec.internal_name,
            "random_state": int(self.config.random_state),
            "net_train_seed": int(self.config.random_state),
            "train_index": fit_scores.index,
            "validation_index": validation_scores.index,
            "safe_feature_transformer": transformer,
        }
        if spec.family != "sklearn":
            train_kwargs.update(
                max_epochs=int(self.config.dl.max_epochs),
                patience=int(self.config.dl.patience),
                device=self.config.device,
                use_amp=bool(self.config.dl.amp),
                batch_size=self.config.dl.batch_size,
                num_workers=int(self.config.dl.num_workers),
            )
        else:
            train_kwargs.update(
                sample_weight=formal_rf_sample_weights(training_scores),
                high_rna_negative_weight=FORMAL_HIGH_RNA_NEGATIVE_WEIGHT,
            )
        trained = train_predict_diagnostic_model(**train_kwargs)
        summary = dict(trained.get("summary", {}))
        if summary.get("status") != "success" or trained.get("fitted_backend") is None:
            raise RuntimeError(f"DuoDose {self.config.backend!r} training failed: {summary.get('message', 'unknown error')}")
        self.backend_ = trained["fitted_backend"]
        self.training_summary_ = summary
        self.feature_audit_ = {
            "included_feature_names": [name for name in str(summary.get("feature_list", "")).split(",") if name],
            "excluded_unsafe_feature_names": [name for name in str(summary.get("unsafe_feature_list", "")).split(",") if name],
            "unsafe_feature_detected": bool(summary.get("unsafe_feature_detected", False)),
            "safe_feature_mode": semireal_config.safe_feature_mode,
            "safe_feature_transformer_id": transformer.transformer_id_,
            "safe_feature_reference_pool_id": transformer.reference_pool_id_,
        }
        self.parent_audit_ = dict(bundle.parent_audit)
        self.construction_report_ = dict(bundle.construction_report)
        self.safe_feature_transformer_ = transformer
        LOGGER.info(
            "DuoDose fit complete: backend=%s transformer=%s",
            self.config.backend,
            transformer.transformer_id_,
        )
        return self

    def predict(self, adata) -> DuoDoseResult:
        if self.backend_ is None:
            raise RuntimeError("DuoDose is not fitted. Call fit or fit_predict first.")
        validate_adata(
            adata,
            layer=self.config.layer,
            expected_doublet_rate=self.config.expected_doublet_rate,
            device=self.config.device,
        )
        work = counts_copy(adata, layer=self.config.layer)
        work.obs["experimental_doublet"] = 0
        if self.safe_feature_transformer_ is None:
            raise RuntimeError("DuoDose fitted backend is missing its fitted-reference SafeFeature transformer")
        cell_scores = self.safe_feature_transformer_.transform(
            work,
            dataset_id="duodose_api_predict",
            random_state=int(self.config.random_state),
        )
        probabilities = self.backend_.predict_probabilities(cell_scores).reindex(work.obs_names)
        from .net import probabilities_to_scores

        overall, _, _ = probabilities_to_scores(probabilities)
        threshold = resolve_threshold(
            overall,
            strategy=self.config.threshold_strategy,
            expected_doublet_rate=float(self.config.expected_doublet_rate),
            probability_threshold=float(self.config.threshold),
        )
        scores = probability_scores_frame(probabilities, threshold)
        LOGGER.info("DuoDose prediction complete: backend=%s cells=%d", self.config.backend, adata.n_obs)
        spec = get_backend_spec(self.config.backend)
        return DuoDoseResult(
            scores=scores,
            threshold=threshold,
            backend=self.config.backend,
            training_summary=dict(self.training_summary_),
            feature_audit=dict(self.feature_audit_),
            parent_audit=dict(self.parent_audit_),
            config=self.config.to_dict(),
            model_metadata={
                "internal_method_name": spec.internal_name,
                "public_method_name": spec.display_name,
                "family": spec.family,
                "role": spec.role,
                "construction_variant": self.construction_report_.get("construction_variant"),
                "safe_feature_mode": self.feature_audit_.get("safe_feature_mode"),
                "parent_disjoint": self.parent_audit_.get("parent_leakage_audit_status") == "passed",
                "high_rna_negative_weight": FORMAL_HIGH_RNA_NEGATIVE_WEIGHT,
            },
        )

    def fit_predict(self, adata) -> DuoDoseResult:
        return self.fit(adata).predict(adata)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return self.config.to_dict()

    def set_params(self, **params: Any) -> "DuoDose":
        for name, value in params.items():
            if not hasattr(self.config, name):
                raise ValueError(f"Unknown DuoDose parameter {name!r}")
            setattr(self.config, name, value)
        self.config.__post_init__()
        self.backend_ = None
        self.safe_feature_transformer_ = None
        return self


def detect_doublets(
    adata,
    *,
    backend: str = DEFAULT_DUODOSE_BACKEND,
    expected_doublet_rate: float = 0.08,
    **kwargs: Any,
) -> DuoDoseResult:
    return DuoDose(backend=backend, expected_doublet_rate=expected_doublet_rate, **kwargs).fit_predict(adata)
