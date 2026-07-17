import numpy as np
import pandas as pd

from duodose.safe_feature_transformer import SafeFeatureTransformer


def test_fitted_reference_is_order_and_chunk_invariant(protocol_adata) -> None:
    reference = protocol_adata[:120, :].copy()
    query = protocol_adata[120:, :].copy()
    transformer = SafeFeatureTransformer(
        random_state=0,
        n_components=8,
        n_clusters=5,
        n_neighbors=10,
        n_artificial_doublets=40,
    ).fit(reference, reference_pool_id="test_reference", dataset="test")

    full = transformer.build_model_matrix(transformer.transform(query, dataset_id="query"))
    order = np.random.default_rng(0).permutation(query.n_obs)
    permuted_query = query[order, :].copy()
    permuted = transformer.build_model_matrix(transformer.transform(permuted_query, dataset_id="query")).reindex(full.index)
    chunks = []
    for positions in np.array_split(np.arange(query.n_obs), 3):
        chunk = query[positions, :].copy()
        chunks.append(transformer.build_model_matrix(transformer.transform(chunk, dataset_id="query")))
    chunked = pd.concat(chunks, axis=0).reindex(full.index)

    assert list(full.columns) == list(transformer.model_feature_columns_)

    score_frame = transformer.transform(query, dataset_id="query")
    expected_balance = np.log1p(score_frame["nFeature"].to_numpy()) - 0.5 * np.log1p(score_frame["nCount"].to_numpy())
    np.testing.assert_allclose(score_frame["library_complexity_balance"].to_numpy(), expected_balance, rtol=0.0, atol=1e-12)
    assert "library_complexity_balance" in full.columns
    np.testing.assert_allclose(full.to_numpy(), permuted.to_numpy(), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(full.to_numpy(), chunked.to_numpy(), rtol=0.0, atol=1e-12)
