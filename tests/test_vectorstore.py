import pytest

from src.utils import IndexNotFoundError
from src.vectorstore import index_exists, load_vector_store


def test_missing_index_is_detected_before_embeddings_are_used(tmp_path) -> None:
    assert not index_exists(tmp_path)
    with pytest.raises(IndexNotFoundError, match="No complete vector index"):
        load_vector_store(tmp_path, embeddings=object())

