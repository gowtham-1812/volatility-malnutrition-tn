import pandas as pd
import numpy as np
import math
from src.features import series_metrics

def test_constant_series():
    # T1 constant [100,100,100] on 3 consecutive months: vol_logret_std == 0, cv == 0, spike_share == 0, range_ratio == 0
    idx = pd.period_range(start="2020-01", periods=3, freq="M")
    series = pd.Series([100, 100, 100], index=idx)
    metrics = series_metrics(series, min_months=2, min_pairs=2)
    
    assert metrics["valid"] == True
    assert metrics["n_months"] == 3
    assert metrics["n_pairs"] == 2
    assert math.isclose(metrics["vol_logret_std"], 0, abs_tol=1e-5)
    assert math.isclose(metrics["cv"], 0, abs_tol=1e-5)
    assert math.isclose(metrics["spike_share"], 0, abs_tol=1e-5)
    assert math.isclose(metrics["range_ratio"], 0, abs_tol=1e-5)

def test_varying_series():
    # T2 [100,200,100,200] on 4 consecutive months
    idx = pd.period_range(start="2020-01", periods=4, freq="M")
    series = pd.Series([100, 200, 100, 200], index=idx)
    metrics = series_metrics(series, min_months=2, min_pairs=2)
    
    assert metrics["valid"] == True
    assert metrics["n_months"] == 4
    assert metrics["n_pairs"] == 3
    assert math.isclose(metrics["vol_logret_std"], 0.800377, abs_tol=1e-5)
    assert math.isclose(metrics["cv"], 0.384900, abs_tol=1e-5)
    assert math.isclose(metrics["spike_share"], 1.0, abs_tol=1e-5)
    assert math.isclose(metrics["range_ratio"], 0.666667, abs_tol=1e-5)

def test_gap_series():
    # T3 gap test: values [100,110,120,130] on months 1, 2, 4, 5 of the same year
    idx = pd.PeriodIndex(["2020-01", "2020-02", "2020-04", "2020-05"], freq="M")
    series = pd.Series([100, 110, 120, 130], index=idx)
    metrics = series_metrics(series, min_months=2, min_pairs=2)
    
    assert metrics["n_months"] == 4
    assert metrics["n_pairs"] == 2

def test_validity_threshold():
    # T4 validity: T2 series with min_months=5 returns valid == False and NaN metrics
    idx = pd.period_range(start="2020-01", periods=4, freq="M")
    series = pd.Series([100, 200, 100, 200], index=idx)
    metrics = series_metrics(series, min_months=5, min_pairs=2)
    
    assert metrics["valid"] == False
    assert np.isnan(metrics["vol_logret_std"])
    assert np.isnan(metrics["cv"])
    assert np.isnan(metrics["spike_share"])
    assert np.isnan(metrics["range_ratio"])

if __name__ == "__main__":
    test_constant_series()
    test_varying_series()
    test_gap_series()
    test_validity_threshold()
    print("All tests passed.")
