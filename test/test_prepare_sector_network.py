# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Endogenous industry process heat: component names and electricity wiring."""

from types import SimpleNamespace

import pandas as pd
import pypsa

from scripts._helpers import check_unique_component_names
from scripts.prepare_sector_network import (
    add_t_industry100_200,
    add_t_industry200_500,
    add_t_industry500,
    insert_electricity_distribution_grid,
)

NODES = pd.Index(["N0"])

COST_ROWS = [
    "solid biomass boiler steam",
    "solid biomass boiler steam CC",
    "gas boiler steam",
    "electric boiler steam",
    "industrial heat pump high temperature",
    "direct firing gas",
    "direct firing gas CC",
    "direct firing solid fuels",
    "direct firing solid fuels CC",
    "biomass CHP capture",
    "biomass boiler",
    "solid biomass",
    "gas",
    "electricity distribution grid",
    "solar-utility",
    "home battery storage",
    "home battery inverter",
    "battery storage",
    "battery inverter",
]


def _costs() -> pd.DataFrame:
    columns = [
        "efficiency",
        "capital_cost",
        "marginal_cost",
        "lifetime",
        "CO2 intensity",
        "capture_rate",
        "heat-input",
        "pelletizing cost",
    ]
    costs = pd.DataFrame(0.0, index=COST_ROWS, columns=columns)
    costs["efficiency"] = 1.0
    costs["lifetime"] = 20.0
    costs.loc[["solid biomass", "gas"], "CO2 intensity"] = 0.2
    costs.at["biomass CHP capture", "capture_rate"] = 0.9
    return costs


def _options() -> dict:
    return {
        "industry_t": {
            "heat100-200": {
                "biomass": True,
                "methane": True,
                "heat_pumps": True,
                "electric_boiler": True,
            },
            "heat200-500": {"biomass": True, "methane": True, "hydrogen": True},
            "heat500+": {"methane": True, "hydrogen": True},
        },
        "transmission_efficiency": {"enable": []},
    }


def _spatial() -> SimpleNamespace:
    return SimpleNamespace(
        biomass=SimpleNamespace(nodes=pd.Index(["EU solid biomass"])),
        gas=SimpleNamespace(nodes=pd.Index(["EU gas"])),
        co2=SimpleNamespace(nodes=pd.Index(["co2 stored"])),
    )


def _network_with_bands() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(["now"])
    n.add("Bus", "N0", carrier="AC")
    n.add("Bus", "N0 H2", carrier="H2")
    demand = pd.DataFrame(
        {"heat100-200": [8760.0], "heat200-500": [8760.0], "heat500+": [8760.0]},
        index=NODES,
    )
    costs, options, spatial = _costs(), _options(), _spatial()
    add_t_industry100_200(n, NODES, demand, costs, 0.8, options, spatial)
    add_t_industry200_500(n, NODES, demand, costs, 0.8, options, spatial)
    add_t_industry500(n, NODES, demand, costs, 0.8, options, spatial)
    return n


def test_industry_heat_band_components_have_unique_names():
    n = _network_with_bands()

    check_unique_component_names(n)

    for band in ("heat100-200", "heat200-500", "heat500+"):
        bus = f"N0 {band} industry"
        assert bus in n.buses.index
        load = f"{bus} demand"
        assert n.loads.at[load, "bus"] == bus
        assert n.loads.at[load, "carrier"] == f"{band} industry"
        assert load not in n.buses.index
    # every supply link is named "<node> <band> industry <technology>"
    links = n.links.index[n.links.bus1.str.endswith(" industry")]
    assert not links.empty
    assert all(
        name == f"N0 {carrier}"
        for name, carrier in n.links.loc[links, "carrier"].items()
    )


def test_distribution_grid_leaves_industrial_heat_pump_on_the_ac_node():
    n = _network_with_bands()
    # a buildings heat pump in the usual reverse orientation, electricity on bus1
    n.add("Bus", "N0 residential rural heat", carrier="residential rural heat")
    n.add(
        "Link",
        "N0 residential rural air heat pump",
        bus0="N0 residential rural heat",
        bus1="N0",
        carrier="residential rural air heat pump",
    )

    insert_electricity_distribution_grid(
        n, _costs(), _options(), pd.DataFrame(index=NODES), ""
    )

    hp = "N0 heat100-200 industry industrial heat pump high temperature"
    boiler = "N0 heat100-200 industry electric boiler steam"
    assert n.links.at[hp, "bus0"] == "N0"
    assert n.links.at[hp, "bus1"] == "N0 heat100-200 industry"
    assert n.links.at[boiler, "bus0"] == "N0"
    assert n.links.at["N0 residential rural air heat pump", "bus1"] == "N0 low voltage"
