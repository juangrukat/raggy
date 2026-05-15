from raggy_mcp.ingest.document_id import (
    compute_document_id,
    compute_source_id,
    compute_version_id,
)


def test_document_id_uses_normalized_content_not_path(tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "nested" / "renamed.md"
    second.parent.mkdir()

    text = "Title\n\n  Same content   \n"

    assert compute_document_id(text, path=str(first)) == compute_document_id(
        "Title\nSame content\n", path=str(second)
    )
    assert compute_source_id(str(first)) != compute_source_id(str(second))


def test_version_id_changes_with_metadata_timestamp():
    text = "same content"

    assert compute_version_id(text, "2026-01-01T00:00:00Z") != compute_version_id(
        text, "2026-01-02T00:00:00Z"
    )
