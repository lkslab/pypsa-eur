# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""The waste sector: HVC and municipal solid waste on waste buses (port of #1654)."""

from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

from scripts._helpers import check_unique_component_names
from scripts.prepare_sector_network import add_waste

NODES = pd.Index(["N0", "N1"])


def _costs() -> pd.DataFrame:
    rows = ["oil", "waste CHP", "waste CHP CC", "biomass CHP capture"]
    columns = [
        "CO2 intensity",
        "capital_cost",
        "VOM",
        "efficiency",
        "efficiency-heat",
        "lifetime",
    ]
    costs = pd.DataFrame(0.0, index=rows, columns=columns)
    costs.at["oil", "CO2 intensity"] = 0.26
    costs.loc[["waste CHP", "waste CHP CC"], "efficiency"] = 0.2
    costs.loc[["waste CHP", "waste CHP CC"], "efficiency-heat"] = 0.76
    costs.loc[["waste CHP", "waste CHP CC"], "capital_cost"] = 100.0
    costs["lifetime"] = 25.0
    return costs


def _spatial() -> SimpleNamespace:
    return SimpleNamespace(
        nodes=NODES,
        waste=SimpleNamespace(
            msw=NODES + " municipal solid waste",
            non_sequestered_hvc=NODES + " non-sequestered HVC",
            buses=NODES + " waste",
            locations=NODES,
            df=pd.DataFrame(
                {
                    "msw": NODES + " municipal solid waste",
                    "non_sequestered_hvc": NODES + " non-sequestered HVC",
                    "buses": NODES + " waste",
                    "locations": NODES,
                },
                index=NODES,
            ),
        ),
        co2=SimpleNamespace(nodes=["co2 stored"]),
    )


def _run(tmp_path, waste_options: dict) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for node in NODES:
        n.add("Bus", node, carrier="AC")
    n.add("Bus", "N0 urban central heat", carrier="urban central heat")
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Bus", "co2 stored", carrier="co2 stored")

    potentials = pd.DataFrame({"municipal solid waste": [3.0, 1.0]}, index=NODES)
    potentials_fn = tmp_path / "biomass_potentials.csv"
    potentials.to_csv(potentials_fn)
    demand = pd.DataFrame(
        {
            ("exogenous", "naphtha"): [10.0, 0.0],
            ("exogenous", "process emission from feedstock"): [0.5, 0.0],
            ("endogenous", "naphtha"): [10.0, 0.0],
            ("endogenous", "process emission from feedstock"): [0.5, 0.0],
        },
        index=NODES,
    )
    demand_fn = tmp_path / "industrial_energy_demand.csv"
    demand.to_csv(demand_fn)
    pop_layout = pd.DataFrame({"total": [3.0, 1.0]}, index=NODES)

    add_waste(
        n,
        _costs(),
        waste_options=waste_options,
        options={"HVC_demand_factor": 1.0, "cc_fraction": 0.9, "waste_spatial": True},
        cf_industry={"HVC_environment_sequestration_fraction": 0.2},
        spatial=_spatial(),
        pop_layout=pop_layout,
        biomass_potentials_file=str(potentials_fn),
        biomass_transport_costs_file="",
        industrial_demand_file=str(demand_fn),
        investment_year=2030,
    )
    return n


def test_hvc_waste_is_distributed_by_population_and_must_be_burnt(tmp_path):
    n = _run(
        tmp_path,
        {"transport": False, "waste_to_energy": False, "waste_to_energy_cc": False},
    )

    check_unique_component_names(n)
    hvc = n.generators[n.generators.carrier == "non-sequestered HVC"]
    # 10 TWh naphtha, 0.8 not sequestered, (0.26 - 0.05) / 0.26 of it is HVC energy
    expected = 10e6 * 0.8 * (0.26 - 0.05) / 0.26
    assert hvc.e_sum_max.sum() == pytest.approx(expected)
    assert hvc.e_sum_min.sum() == pytest.approx(expected)
    # three quarters of the people live at N0
    assert hvc.at["N0 non-sequestered HVC", "e_sum_max"] == pytest.approx(
        0.75 * expected
    )
    assert n.links.at["N0 waste to air", "efficiency"] == pytest.approx(0.26)
    assert "municipal solid waste" not in n.carriers.index


def test_waste_to_energy_brings_msw_and_chps_on_the_waste_bus(tmp_path):
    n = _run(
        tmp_path,
        {"transport": False, "waste_to_energy": True, "waste_to_energy_cc": True},
    )

    check_unique_component_names(n)
    msw = n.generators.loc["N0 municipal solid waste Generator"]
    assert msw.e_sum_max == pytest.approx(3.0)
    bridge = n.links.loc["N0 municipal solid waste to waste"]
    assert bridge.bus1 == "N0 waste" and bridge.efficiency2 == pytest.approx(-0.26)
    chp = n.links.loc["N0 waste CHP"]
    assert chp.bus0 == "N0 waste" and chp.bus2 == "N0 urban central heat"
    assert n.links.at["N1 waste CHP", "bus2"] == ""
    cc = n.links.loc["N0 waste CHP CC"]
    assert cc.bus4 == "co2 stored"
    assert cc.efficiency4 == pytest.approx(0.26 * 0.9)
