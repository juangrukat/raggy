from raggy_mcp.search.filter_grammar import compile_filter


def test_compile_filter_prefixes_metadata_fields():
    qfilter = compile_filter(
        {"must": [{"field": "extension", "op": "==", "value": "pdf"}]}
    )

    assert qfilter.must[0].key == "metadata.extension"
    assert qfilter.must[0].match.value == "pdf"


def test_compile_filter_range_and_any():
    qfilter = compile_filter(
        {
            "must": [{"field": "size_bytes", "op": ">=", "value": 100}],
            "should": [{"field": "tags", "op": "any", "value": ["work", "urgent"]}],
        }
    )

    assert qfilter.must[0].range.gte == 100
    assert qfilter.should[0].match.any == ["work", "urgent"]


def test_compile_filter_not_equal_embeds_must_not_filter():
    qfilter = compile_filter(
        {"must": [{"field": "is_hidden", "op": "!=", "value": True}]}
    )

    nested = qfilter.must[0]
    assert nested.must_not[0].key == "metadata.is_hidden"
    assert nested.must_not[0].match.value is True


def test_compile_empty_filter_returns_none():
    assert compile_filter({}) is None
    assert compile_filter({"must": []}) is None


def test_compile_filter_reuses_cached_equivalent_specs():
    spec_a = {"must": [{"field": "extension", "op": "==", "value": "pdf"}]}
    spec_b = {"must": [{"value": "pdf", "op": "==", "field": "extension"}]}

    assert compile_filter(spec_a) is compile_filter(spec_b)
