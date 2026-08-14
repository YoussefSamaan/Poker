from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = (
    "street",
    "position",
    "previous_aggressor_relationship",
)


def build_preprocessor(feature_names: Sequence[str]) -> ColumnTransformer:
    categorical = [
        index
        for index, name in enumerate(feature_names)
        if name in CATEGORICAL_FEATURES
        or name.endswith("hand_class")
        or name.endswith("made_category")
    ]
    numeric = [index for index in range(len(feature_names)) if index not in categorical]
    return ColumnTransformer(
        (
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        )
    )


def feature_matrix(
    rows: Sequence[Mapping[str, object]], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[row.get(name, 0.0) for name in feature_names] for row in rows],
        dtype=object,
    )


def transformed_feature_names(
    pipeline: Pipeline, input_features: Sequence[str]
) -> tuple[str, ...]:
    processor = pipeline.named_steps["preprocess"]
    return tuple(map(str, processor.get_feature_names_out(input_features)))
