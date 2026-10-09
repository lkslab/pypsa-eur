# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Endogenous shipping fuel choice and LNG shipping (port of PyPSA/pypsa-eur#2317)."""

from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

from scripts._helpers import check_unique_component_names
from scripts.prepare_sector_network import add_shipping

NODES = pd.Index(["N0", "N1"])


def _costs() -> pd.DataFrame:
    rows = [
        "CH4 liquefaction",
        "H2 liquefaction",
        "fuel cell",
        "gas",
        "oil",
        "methanolisation",
    ]
    columns = [
        "efficiency",
        "capital_cost",
        "lifetime",
        "electricity-input",
        "CO2 intensity",
        "carbondioxide-input",
    ]
    costs = pd.DataFrame(0.0, index=rows, columns=columns)
    costs["efficiency"] = 1.0
    costs["lifetime"] = 25.0
    costs.at["fuel cell", "efficiency"] = 0.5
    costs.at["CH4 liquefaction", "electricity-input"] = 0.036
    costs.at["CH4 liquefaction", "capital_cost"] = 30.0
    costs.at["gas", "CO2 intensity"] = 0.2
    costs.at["oil", "CO2 intensity"] = 0.26
    costs.at["methanolisation", "carbondioxide-input"] = 0.25
    return costs


def _options(**overrides) -> dict:
    options = {
        "shipping_endogenous": True,
        "shipping_oil": True,
        "shipping_methanol": True,
        "shipping_lng": True,
        "shipping_hydrogen": {2030: False, 2040: True},
        "shipping_hydrogen_liquefaction": False,
        "shipping_hydrogen_share": {2030: 0.0, 2040: 0.0},
        "shipping_methanol_share": {2030: 0.5, 2040: 0.5},
        "shipping_oil_share": {2030: 0.3, 2040: 0.3},
        "shipping_lng_share": {2030: 0.2, 2040: 0.2},
        "shipping_oil_efficiency": 0.4,
        "shipping_methanol_efficiency": 0.46,
        "shipping_lng_efficiency": 0.45,
        "shipping_lng_methane_slip": 0.031,
        "shipping_methane_gwp100": 25.0,
        "methanol": {"regional_methanol_demand": False},
        "regional_oil_demand": False,
    }
    options.update(overrides)
    return options


def _spatial() -> SimpleNamespace:
    return SimpleNamespace(
        gas=SimpleNamespace(df=pd.DataFrame({"nodes": "EU gas"}, index=NODES)),
        oil=SimpleNamespace(
            nodes=["EU oil"], shipping=["EU shipping oil"], demand_locations=["EU"]
        ),
        methanol=SimpleNamespace(
            nodes=["EU methanol"],
            shipping=["EU shipping methanol"],
            demand_locations=["EU"],
        ),
    )


def _network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for node in NODES:
        n.add("Bus", node, carrier="AC")
        n.add("Bus", f"{node} H2", carrier="H2")
    for bus, carrier in [
        ("EU gas", "gas"),
        ("EU oil", "oil"),
        ("EU methanol", "methanol"),
    ]:
        n.add("Bus", bus, carrier=carrier)
    n.add("Bus", "co2 atmosphere", carrier="co2")
    return n


def _run(options: dict, year: int, tmp_path) -> pypsa.Network:
    n = _network()
    demand = pd.Series([10.0, 20.0], index=NODES, name="international navigation")
    fn = tmp_path / "shipping_demand.csv"
    demand.to_csv(fn)
    totals = pd.DataFrame({"total domestic navigation": [2.0, 4.0]}, index=NODES)
    add_shipping(
        n,
        _costs(),
        str(fn),
        pd.DataFrame(index=NODES),
        totals,
        options,
        _spatial(),
        year,
    )
    return n


def test_endogenous_shipping_has_one_demand_bus_and_competing_links(tmp_path):
    n = _run(_options(), 2040, tmp_path)

    check_unique_component_names(n)
    assert n.loads.at["N0 shipping demand", "bus"] == "N0 shipping"
    # domestic navigation (already per year) plus the international demand scaled
    # to the two-hour network, in oil-equivalent energy
    annual = 2.0 + 10.0 * 2 / 8760
    assert n.loads.at["N0 shipping demand", "p_set"] == pytest.approx(annual * 1e6 / 2)
    links = n.links[n.links.bus1 == "N0 shipping"]
    assert set(links.carrier) == {
        "shipping oil",
        "shipping methanol",
        "shipping LNG",
        "H2 for shipping",
    }
    assert n.links.at["N0 shipping oil", "efficiency"] == pytest.approx(1.0)
    assert n.links.at["N0 shipping methanol", "efficiency"] == pytest.approx(0.46 / 0.4)
    assert n.links.at["N0 H2 for shipping", "efficiency"] == pytest.approx(0.5 / 0.4)
    lng = n.links.loc["N0 shipping LNG"]
    assert lng.bus0 == "N0 LNG"
    assert lng.efficiency == pytest.approx(0.45 * (1 - 0.031) / 0.4)
    assert lng.efficiency2 == pytest.approx((1 - 0.031) * 0.2 + 0.031 * 0.072 * 25.0)
    liquefaction = n.links.loc["N0 CH4 liquefaction"]
    assert liquefaction.bus0 == "EU gas" and liquefaction.bus1 == "N0 LNG"
    assert liquefaction.efficiency2 == pytest.approx(-0.036)


def test_year_indexed_fuel_switch_and_no_fuel_raises(tmp_path):
    n = _run(_options(), 2030, tmp_path)
    assert "N0 H2 for shipping" not in n.links.index

    with pytest.raises(ValueError, match="At least one"):
        _run(
            _options(
                shipping_oil=False,
                shipping_methanol=False,
                shipping_lng=False,
                shipping_hydrogen=False,
            ),
            2030,
            tmp_path,
        )


def test_exogenous_lng_share_adds_a_liquefied_demand_chain(tmp_path):
    n = _run(_options(shipping_endogenous=False), 2030, tmp_path)

    check_unique_component_names(n)
    assert "N0 shipping" not in n.buses.index
    load = n.loads.loc["N0 shipping LNG demand"]
    assert load.bus == "N0 shipping LNG"
    useful = 0.45 * (1 - 0.031)
    annual = 2.0 + 10.0 * 2 / 8760
    assert load.p_set == pytest.approx(0.2 * annual * 1e6 / 2 * 0.4 / useful)
    conversion = n.links.loc["N0 shipping LNG conversion"]
    assert conversion.bus0 == "N0 LNG" and conversion.bus1 == "N0 shipping LNG"
    assert "N0 CH4 liquefaction" in n.links.index
