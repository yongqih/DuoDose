import pandas as pd

from duodose.result import DuoDoseResult


def test_result_adds_expected_obs_columns(small_adata) -> None:
    frame = pd.DataFrame(
        {
            "duodose_score": 0.2,
            "duodose_homotypic_score": 0.1,
            "duodose_heterotypic_score": 0.1,
            "predicted_doublet": False,
            "predicted_subtype": "",
            "subtype_confidence": float("nan"),
        },
        index=small_adata.obs_names,
    )
    result = DuoDoseResult(frame, 0.5, "dl")
    result.add_to_adata(small_adata)
    assert "duodose_prediction" in small_adata.obs
    assert small_adata.uns["duodose"]["backend"] == "dl"
