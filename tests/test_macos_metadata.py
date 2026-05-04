from raggy_mcp.ingest.macos_metadata import (
    get_macos_metadata,
    get_macos_metadata_async,
)


async def test_async_metadata_preserves_base_fields(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")

    sync_meta = get_macos_metadata(str(path))
    async_meta = await get_macos_metadata_async(str(path))

    for key in ("path", "filename", "extension", "size_bytes", "is_hidden", "has_text", "tags"):
        assert async_meta[key] == sync_meta[key]
