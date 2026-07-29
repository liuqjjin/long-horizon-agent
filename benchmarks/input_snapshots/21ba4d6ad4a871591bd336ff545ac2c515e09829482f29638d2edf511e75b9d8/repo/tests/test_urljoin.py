from urljoin import join_url


def test_simple_join():
    assert join_url("https://x.io", "a", "b") == "https://x.io/a/b"


def test_base_trailing_slash_not_doubled():
    assert join_url("https://api.example.com/", "v1") == "https://api.example.com/v1"


def test_part_leading_slash_not_doubled():
    assert join_url("https://api.example.com", "/v1") == "https://api.example.com/v1"


def test_slashes_on_both_sides_of_a_boundary():
    assert join_url("https://api.example.com/", "/v1", "/users") == (
        "https://api.example.com/v1/users"
    )


def test_scheme_separator_survives():
    out = join_url("https://x.io/", "/a")
    assert out.startswith("https://")
    assert out == "https://x.io/a"


def test_base_with_existing_path():
    assert join_url("http://x.io/base/", "y") == "http://x.io/base/y"


def test_empty_parts_are_skipped():
    assert join_url("https://x.io", "", "a") == "https://x.io/a"


def test_part_with_internal_path_kept():
    assert join_url("https://x.io", "a/b/c") == "https://x.io/a/b/c"


def test_trailing_slash_of_last_part_survives():
    assert join_url("https://x.io", "a/") == "https://x.io/a/"


def test_no_parts_returns_base():
    assert join_url("https://x.io") == "https://x.io"
