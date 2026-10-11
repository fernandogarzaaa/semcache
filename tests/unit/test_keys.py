from semcache.keys import (
    PARAM_FIELDS,
    build_key,
    canonical_json,
    has_non_text_content,
    is_cacheable,
    last_user_index,
    message_text,
    sha256_hex,
    system_hash,
    system_prompt,
)


def body(user="What is Python?", system=None, **kw):
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return {"model": "gpt-4o-mini", "messages": msgs, **kw}


def test_canonical_json_is_order_independent_and_compact():
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert canonical_json({"x": "é"}) == '{"x":"é"}'


def test_sha256_hex_known_value():
    assert sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_message_text_variants():
    assert message_text({"content": "hi"}) == "hi"
    assert message_text({"content": None}) == ""
    assert message_text({}) == ""
    parts = {"content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, "b"]}
    assert message_text(parts) == "a\nb"
    assert message_text({"content": [{"type": "text"}]}) == ""
    assert message_text({"content": 42}) == "42"


def test_has_non_text_content():
    assert not has_non_text_content([{"content": "x"}])
    assert not has_non_text_content([{"content": [{"type": "text", "text": "x"}]}])
    assert has_non_text_content([{"content": [{"type": "text", "text": "x"}, {"type": "image_url"}]}])
    assert not has_non_text_content([{"content": ["plain string part"]}])


def test_last_user_index():
    msgs = [{"role": "user"}, {"role": "assistant"}, {"role": "user"}, {"role": "assistant"}]
    assert last_user_index(msgs) == 2
    assert last_user_index([{"role": "system"}]) == -1
    assert last_user_index([]) == -1
    assert last_user_index([{"role": "user"}]) == 0


def test_system_prompt_joins_system_and_developer_in_order():
    msgs = [
        {"role": "system", "content": "A"},
        {"role": "user", "content": "q"},
        {"role": "developer", "content": "B"},
    ]
    assert system_prompt(msgs) == "A\nB"
    assert system_hash(msgs) == sha256_hex("A\nB")[:16]
    assert len(system_hash(msgs)) == 16


def test_build_key_none_without_user_message():
    assert build_key({"model": "m", "messages": [{"role": "system", "content": "x"}]}) is None
    assert build_key({"model": "m"}) is None


def test_build_key_fields():
    k = build_key(body(system="be brief"))
    assert k is not None
    assert k.query == "What is Python?"
    assert k.model == "gpt-4o-mini"
    assert k.system_hash == sha256_hex("be brief")[:16]
    assert len(k.namespace) == 32


def test_same_context_different_question_share_namespace():
    a = build_key(body("What is Python?"))
    b = build_key(body("Explain Python to me"))
    assert a.namespace == b.namespace
    assert a.query != b.query


def test_system_prompt_model_and_params_split_namespaces():
    base = build_key(body())
    assert build_key(body(system="legal")).namespace != base.namespace
    assert build_key({**body(), "model": "gpt-4o"}).namespace != base.namespace
    for field, value in [("temperature", 0.7), ("max_tokens", 50), ("top_p", 0.5), ("seed", 1), ("tools", [{}])]:
        assert build_key(body(**{field: value})).namespace != base.namespace, field


def test_irrelevant_fields_do_not_change_namespace():
    base = build_key(body())
    assert build_key(body(stream=True)).namespace == base.namespace
    assert build_key({**body(), "metadata": {"a": 1}}).namespace == base.namespace
    assert build_key(body(temperature=None)).namespace == base.namespace


def test_history_is_part_of_namespace():
    one = {"model": "m", "messages": [{"role": "user", "content": "and in Java?"}]}
    two = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "how to reverse a list in Python"},
            {"role": "assistant", "content": "use reversed()"},
            {"role": "user", "content": "and in Java?"},
        ],
    }
    assert build_key(one).namespace != build_key(two).namespace
    assert build_key(two).query == "and in Java?"


def test_param_fields_cover_sampling_controls():
    for f in ("temperature", "top_p", "max_tokens", "stop", "seed", "tools", "response_format"):
        assert f in PARAM_FIELDS
    assert "stream" not in PARAM_FIELDS
    assert "messages" not in PARAM_FIELDS


def test_is_cacheable():
    assert is_cacheable(body())
    assert is_cacheable(body(n=1))
    assert not is_cacheable(body(n=2))
    img = {"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}
    assert not is_cacheable(img)
    assert is_cacheable({"model": "m"})
