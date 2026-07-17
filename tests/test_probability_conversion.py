import pandas as pd
import pytest

from duodose.net import probabilities_to_scores


def test_probability_conversion() -> None:
    probabilities = pd.DataFrame(
        {
            "clean": [0.6, 0.1],
            "high_RNA_singlet": [0.1, 0.1],
            "homotypic_doublet": [0.2, 0.3],
            "heterotypic_doublet": [0.1, 0.5],
        }
    )
    overall, homotypic, heterotypic = probabilities_to_scores(probabilities)
    assert overall.tolist() == pytest.approx([0.3, 0.8])
    assert homotypic.tolist() == pytest.approx([0.2, 0.3])
    assert heterotypic.tolist() == pytest.approx([0.1, 0.5])
