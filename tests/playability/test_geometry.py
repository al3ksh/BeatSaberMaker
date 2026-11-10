import pytest

from bsmaker.playability.geometry import angle_diff, direction_angle


@pytest.mark.parametrize(
    "direction,angle",
    [(0, 90), (1, 270), (2, 180), (3, 0), (4, 135), (5, 45), (6, 225), (7, 315), (8, None)],
)
def test_direction_angle(direction, angle):
    assert direction_angle(direction) == angle


def test_angle_diff_wraps():
    assert angle_diff(350, 10) == 20
    assert angle_diff(10, 350) == 20


def test_angle_diff_opposite():
    assert angle_diff(90, 270) == 180
