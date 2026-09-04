from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import joblib
import sklearn

from .action_model import (
    BoostedContextActionModel,
    BoostedHistoryActionModel,
    ContextActionModel,
    HistoryAwareActionModel,
)
from .schema import FEATURE_SCHEMA_VERSION

ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LearnedArtifactMetadata:
    schema_version: int
    model_type: str
    feature_schema_version: int
    training_dataset_fingerprint: str
    training_rows: int
    training_correlation_groups: int
    sklearn_version: str
    action_classes: tuple[str, ...]
    metrics_summary: Mapping[str, float]
    seed: int


def dataset_fingerprint(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def build_metadata(
    model: object,
    *,
    dataset_payload: str,
    training_rows: int,
    training_correlation_groups: int,
    metrics_summary: Mapping[str, float],
    seed: int,
) -> LearnedArtifactMetadata:
    return LearnedArtifactMetadata(
        ARTIFACT_SCHEMA_VERSION,
        model.MODEL_TYPE,
        FEATURE_SCHEMA_VERSION,
        dataset_fingerprint(dataset_payload),
        training_rows,
        training_correlation_groups,
        sklearn.__version__,
        tuple(model.classes),
        dict(metrics_summary),
        seed,
    )


def save_learned_artifact(
    path: str | Path, model: object, metadata: LearnedArtifactMetadata
) -> None:
    if metadata.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported learned artifact schema")
    if metadata.model_type != model.MODEL_TYPE:
        raise ValueError("artifact metadata does not match model type")
    joblib.dump(
        {
            "metadata": asdict(metadata),
            "pipeline": model.pipeline,
            "feature_names": model.feature_names,
            "seed": model.seed,
        },
        Path(path),
    )


def load_trusted_local_artifact(
    path: str | Path,
) -> tuple[object, LearnedArtifactMetadata]:
    """Load only a trusted local joblib artifact; pickle formats are code-executing."""
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict) or set(payload) != {
        "metadata", "pipeline", "feature_names", "seed"
    }:
        raise ValueError("invalid learned artifact envelope")
    metadata = LearnedArtifactMetadata(**payload["metadata"])
    if metadata.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported learned artifact schema")
    if metadata.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported learned feature schema")
    classes = {
        ContextActionModel.MODEL_TYPE: ContextActionModel,
        HistoryAwareActionModel.MODEL_TYPE: HistoryAwareActionModel,
        BoostedContextActionModel.MODEL_TYPE: BoostedContextActionModel,
        BoostedHistoryActionModel.MODEL_TYPE: BoostedHistoryActionModel,
    }
    if metadata.model_type == "hand_conditioned_logistic":
        from .hand_conditioned import HandConditionedActionModel

        classes[metadata.model_type] = HandConditionedActionModel
    try:
        model_class = classes[metadata.model_type]
    except KeyError as error:
        raise ValueError("unknown learned model type") from error
    model = model_class(seed=payload["seed"])
    model.pipeline = payload["pipeline"]
    model.feature_names = tuple(payload["feature_names"])
    if tuple(model.classes) != tuple(metadata.action_classes):
        raise ValueError("artifact action classes do not match fitted pipeline")
    return model, metadata


def metadata_json(metadata: LearnedArtifactMetadata) -> str:
    return json.dumps(asdict(metadata), indent=2, sort_keys=True)
