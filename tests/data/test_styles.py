from bsmaker.data.styles import STYLES, UNKNOWN, styles_from_tags


def test_styles_from_tags():
    assert styles_from_tags(["tech", "dance-style", "anime"]) == ("tech", "dance")
    assert styles_from_tags([]) == (UNKNOWN,) == ("unknown",)
    assert set(STYLES) >= {"tech", "dance", "fitness"}
