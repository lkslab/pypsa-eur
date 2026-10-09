# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Hourly industry electricity profiles from FfE branch profiles (port of #1875)."""

import numpy as np
import pandas as pd
import pytest

from scripts.build_industrial_energy_demand_per_node import (
    FFE_PROFILE_NAMES,
    FFE_REFERENCE_YEAR,
    INDUSTRY_SECTOR_TO_PROFILE,
    build_electricity_profiles,
    map_profiles_to_snapshots,
)


def _ffe_profiles() -> pd.DataFrame:
    """Synthetic branch profiles: a day-night shape, lower on Sundays."""
    index = pd.date_range(
        f"{FFE_REFERENCE_YEAR}-01-01", f"{FFE_REFERENCE_YEAR}-12-31 23:00", freq="h"
    )
    day = 1.0 + 0.5 * np.sin((index.hour - 6) / 24 * 2 * np.pi)
    sunday = np.where(index.dayofweek == 6, 0.5, 1.0)
    frame = pd.DataFrame(index=index)
    for i, name in enumerate(FFE_PROFILE_NAMES.values()):
        frame[name] = day * sunday * (1 + 0.1 * i)
    return frame.div(frame.sum())


def test_mapped_profiles_sum_to_one_and_keep_weekday_alignment():
    profiles = _ffe_profiles()
    snapshots = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h")

    mapped = map_profiles_to_snapshots(profiles, snapshots, "NL")

    assert mapped.shape == (len(snapshots), len(FFE_PROFILE_NAMES))
    assert np.allclose(mapped.sum(), 1.0)
    assert not mapped.isna().any().any()
    # Sundays stay the low days after the shift onto the 2030 calendar
    by_dow = mapped["Machinery"].groupby(mapped.index.dayofweek).sum()
    assert by_dow[6] < 0.6 * by_dow[0]
    # a Dutch holiday on a weekday (King's Day, Saturday in 2030, so take
    # Christmas, Wednesday 25 Dec 2030) carries the Sunday shape
    christmas = mapped.loc["2030-12-25", "Machinery"].sum()
    wednesday = mapped.loc["2030-12-18", "Machinery"].sum()
    assert christmas < 0.6 * wednesday


def test_mapped_profiles_cover_a_leap_year_and_a_dropped_leap_day():
    profiles = _ffe_profiles()
    full = pd.date_range("2028-01-01", "2028-12-31 23:00", freq="h")
    without = full[~((full.month == 2) & (full.day == 29))]

    for snapshots in (full, without):
        mapped = map_profiles_to_snapshots(profiles, snapshots, "DE")
        assert len(mapped) == len(snapshots)
        assert not mapped.isna().any().any()
        assert np.allclose(mapped.sum(), 1.0)


def test_node_profiles_follow_the_sector_mix_and_fall_back_to_flat():
    profiles = _ffe_profiles()
    snapshots = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h")
    sectors = list(INDUSTRY_SECTOR_TO_PROFILE)
    electricity = pd.DataFrame(0.0, index=["N0", "N1", "N2"], columns=sectors)
    electricity.loc["N0", "Machinery equipment"] = 10.0
    electricity.loc["N1", "Integrated steelworks"] = 5.0
    electricity.loc["N1", "Food, beverages and tobacco"] = 5.0
    countries = pd.Series({"N0": "DE", "N1": "DE", "N2": "DE"})

    node_profiles = build_electricity_profiles(
        electricity, countries, profiles, snapshots
    )

    assert list(node_profiles.columns) == ["N0", "N1", "N2"]
    assert np.allclose(node_profiles.sum(), 1.0)
    mapped = map_profiles_to_snapshots(profiles, snapshots, "DE")
    assert np.allclose(node_profiles["N0"], mapped["Machinery"])
    expected = 0.5 * mapped["Iron & steel industry"] + 0.5 * mapped["Food and Tobacco"]
    assert np.allclose(node_profiles["N1"], expected)
    assert np.allclose(node_profiles["N2"], 1.0 / len(snapshots))


def test_unmapped_sector_is_an_error():
    profiles = _ffe_profiles()
    snapshots = pd.date_range("2030-01-01", periods=24, freq="h")
    electricity = pd.DataFrame({"Quarrying of moon rock": [1.0]}, index=["N0"])

    with pytest.raises(ValueError, match="moon rock"):
        build_electricity_profiles(
            electricity, pd.Series({"N0": "DE"}), profiles, snapshots
        )
