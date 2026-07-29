import pytest
from runstats import OnlineStats


def _fed(values):
    s = OnlineStats()
    for v in values:
        s.add(v)
    return s


def test_no_samples_raises():
    s = OnlineStats()
    with pytest.raises(ValueError):
        _ = s.mean
    with pytest.raises(ValueError):
        _ = s.variance


def test_single_sample():
    s = _fed([7.0])
    assert s.mean == 7.0
    assert s.variance == 0.0


def test_small_case_exact():
    s = _fed([1.0, 2.0, 3.0])
    assert s.mean == pytest.approx(2.0)
    assert s.variance == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_constant_data_has_zero_variance():
    s = _fed([7.0] * 10)
    assert s.variance == pytest.approx(0.0, abs=1e-12)
    assert s.variance >= 0.0


def test_variance_is_never_negative():
    s = _fed([1e9 + 0.001, 1e9 + 0.002, 1e9 + 0.003])
    assert s.variance >= 0.0


def test_large_offset_does_not_change_variance():
    # var([K+1, K+2, K+3]) must equal var([1, 2, 3]) = 2/3.
    k = 1e9
    s = _fed([k + 1, k + 2, k + 3])
    assert s.mean == pytest.approx(k + 2)
    assert s.variance == pytest.approx(2.0 / 3.0, abs=1e-3)


def test_large_offset_longer_stream():
    # Population variance of 0..99 is (100^2 - 1) / 12 = 833.25.
    k = 1e9
    s = _fed([k + i for i in range(100)])
    assert s.variance == pytest.approx(833.25, rel=1e-6)


def test_n_tracks_sample_count():
    assert _fed([1.0, 2.0]).n == 2
