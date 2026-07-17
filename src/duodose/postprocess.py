"""Convert class probabilities into public DuoDose output columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .net import probabilities_to_scores


def probability_scores_frame(probabilities: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    overall, homotypic, heterotypic = probabilities_to_scores(probabilities)
    output = pd.DataFrame(index=probabilities.index)
    output["duodose_score"] = overall
    output["duodose_homotypic_score"] = homotypic
    output["duodose_heterotypic_score"] = heterotypic
    if threshold is None:
        predicted = pd.Series(pd.NA, index=output.index, dtype="boolean")
    else:
        predicted = overall.ge(float(threshold)).astype(bool)
    output["predicted_doublet"] = predicted

    total_subtype = (homotypic + heterotypic).to_numpy(dtype=float)
    subtype_max = np.maximum(homotypic.to_numpy(dtype=float), heterotypic.to_numpy(dtype=float))
    confidence = np.divide(subtype_max, total_subtype, out=np.full(len(output), 0.5), where=total_subtype > 0)
    output["subtype_confidence"] = confidence
    subtype = np.where(
        homotypic.to_numpy(dtype=float) >= heterotypic.to_numpy(dtype=float),
        "homotypic_doublet",
        "heterotypic_doublet",
    ).astype(object)
    if threshold is None:
        subtype[:] = ""
        output["subtype_confidence"] = np.nan
    else:
        subtype[~predicted.to_numpy(dtype=bool)] = ""
        output.loc[~predicted.to_numpy(dtype=bool), "subtype_confidence"] = np.nan
    output["predicted_subtype"] = subtype
    return output[
        [
            "duodose_score",
            "duodose_homotypic_score",
            "duodose_heterotypic_score",
            "predicted_doublet",
            "predicted_subtype",
            "subtype_confidence",
        ]
    ]
