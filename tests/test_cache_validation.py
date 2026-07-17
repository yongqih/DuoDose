import numpy as np
import pandas as pd

from duodose.realdata import _cache_key, _load_cached_score


def _write_cache(tmp_path, adata, *, status: str, scores) -> None:
    key = _cache_key("test", "Scrublet", adata, 0, {"x": 1})
    pd.DataFrame(
        {"cell_id": adata.obs_names, "score": scores, "status": status, "message": "test"}
    ).to_csv(tmp_path / f"test__Scrublet__{key}.csv", index=False)


def test_failed_cache_is_ignored(tmp_path, small_adata) -> None:
    _write_cache(tmp_path, small_adata, status="failed", scores=np.zeros(small_adata.n_obs))
    assert _load_cached_score(tmp_path, "test", "Scrublet", small_adata, 0, False, {"x": 1}) is None


def test_nan_cache_is_ignored(tmp_path, small_adata) -> None:
    _write_cache(tmp_path, small_adata, status="success", scores=np.full(small_adata.n_obs, np.nan))
    assert _load_cached_score(tmp_path, "test", "Scrublet", small_adata, 0, False, {"x": 1}) is None
