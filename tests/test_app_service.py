from dataclasses import replace

from src.app_service import get_health_status
from src.config import Settings


def test_health_status_reports_papers_and_complete_index(tmp_path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "example.pdf").touch()
    index_dir = tmp_path / "vectorstore"
    index_dir.mkdir()
    (index_dir / "index.faiss").touch()
    (index_dir / "index.pkl").touch()

    settings = replace(
        Settings.from_env(), papers_dir=papers_dir, vector_store_dir=index_dir
    )

    health = get_health_status(settings)

    assert health.ready is True
    assert health.papers == ("example.pdf",)


def test_health_status_requires_a_complete_index(tmp_path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "example.pdf").touch()
    index_dir = tmp_path / "vectorstore"
    index_dir.mkdir()
    (index_dir / "index.faiss").touch()

    settings = replace(
        Settings.from_env(), papers_dir=papers_dir, vector_store_dir=index_dir
    )

    assert get_health_status(settings).ready is False
