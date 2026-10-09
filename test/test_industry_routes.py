# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Endogenous steel and cement routes (port of PyPSA/pypsa-eur#1719)."""

from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

from scripts._helpers import check_unique_component_names
from scripts.add_existing_baseyear import add_existing_industry
from scripts.build_industrial_energy_demand_per_node import route_endogenised_sectors
from scripts.prepare_sector_network import add_cement, add_steel

NODES = pd.Index(["N0", "N1"])

COST_ROWS = [
    "blast furnace-basic oxygen furnace",
    "hydrogen direct iron reduction furnace",
    "natural gas direct iron reduction furnace",
    "electric arc furnace",
    "iron ore DRI-ready",
    "steel carbon capture retrofit",
    "cement dry clinker",
    "cement finishing",
    "cement carbon capture retrofit",
    "Haber-Bosch",
    "grey methanol synthesis",
    "SMR",
    "methanolisation",
    "methanol",
    "solid biomass",
    "gas",
    "oil",
    "coal",
    "coal storage",
]


def _costs() -> pd.DataFrame:
    columns = [
        "capital_cost",
        "marginal_cost",
        "VOM",
        "lifetime",
        "efficiency",
        "CO2 intensity",
        "capture_rate",
        "electricity-input",
        "hydrogen-input",
        "gas-input",
        "coal-input",
        "ore-input",
        "hbi-input",
        "heat-input",
        "clinker-input",
        "commodity",
        "fuel",
    ]
    costs = pd.DataFrame(0.0, index=COST_ROWS, columns=columns)
    costs["lifetime"] = 25.0
    costs["efficiency"] = 1.0
    costs.loc[["solid biomass", "gas", "oil", "coal"], "CO2 intensity"] = 0.3
    costs.loc[
        ["steel carbon capture retrofit", "cement carbon capture retrofit"],
        "capture_rate",
    ] = 0.9
    costs.at["blast furnace-basic oxygen furnace", "coal-input"] = 5.34
    costs.at["blast furnace-basic oxygen furnace", "ore-input"] = 1.54
    costs.at["hydrogen direct iron reduction furnace", "electricity-input"] = 1.03
    costs.at["hydrogen direct iron reduction furnace", "hydrogen-input"] = 2.1
    costs.at["hydrogen direct iron reduction furnace", "ore-input"] = 1.59
    costs.at["natural gas direct iron reduction furnace", "gas-input"] = 2.78
    costs.at["electric arc furnace", "electricity-input"] = 0.64
    costs.at["electric arc furnace", "hbi-input"] = 1.0
    costs.at["iron ore DRI-ready", "commodity"] = 97.7
    costs.at["cement dry clinker", "heat-input"] = 0.9444
    costs.at["cement dry clinker", "gas-input"] = 0.0002
    costs.at["cement dry clinker", "electricity-input"] = 0.0694
    costs.at["cement finishing", "clinker-input"] = 0.656
    costs.at["cement finishing", "electricity-input"] = 0.1736
    costs.at["Haber-Bosch", "electricity-input"] = 0.25
    costs.at["Haber-Bosch", "hydrogen-input"] = 5.93
    costs.at["grey methanol synthesis", "efficiency"] = 0.569
    return costs


def _options() -> dict:
    return {
        "endogenous_sectors": {"enable": True, "subsectors": ["steel", "cement"]},
        "hbi_relocation": False,
        "steel_bof": {"pledge": True, "pledge_delay": 0},
        "cement": {"calcination_emissions": 0.5071},
        "ammonia": True,
        "methanol": True,
        "industry": True,
        "fossil_fuels": True,
        "waste": True,
    }


def _spatial() -> SimpleNamespace:
    return SimpleNamespace(
        nodes=NODES,
        hbi=SimpleNamespace(nodes=NODES + " hbi", locations=NODES),
        steel=SimpleNamespace(nodes=NODES + " steel", locations=NODES),
        cement=SimpleNamespace(
            nodes=NODES + " cement",
            heat=NODES + " cement heat",
            emissions=NODES + " cement emission",
            locations=NODES,
        ),
        clinker=SimpleNamespace(nodes=NODES + " clinker", locations=NODES),
        h2=SimpleNamespace(nodes=NODES + " H2"),
        gas=SimpleNamespace(
            nodes=["EU gas"],
            locations=["EU"],
            df=pd.DataFrame({"nodes": "EU gas"}, index=NODES),
        ),
        biomass=SimpleNamespace(
            nodes=["EU solid biomass"],
            df=pd.DataFrame({"nodes": "EU solid biomass"}, index=NODES),
        ),
        waste=SimpleNamespace(df=pd.DataFrame({"buses": "EU waste"}, index=NODES)),
        co2=SimpleNamespace(nodes=["co2 stored"]),
        coal=SimpleNamespace(nodes=["EU coal"], locations=["EU"]),
        methanol=SimpleNamespace(nodes=["EU methanol"]),
    )


def _base_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for node in NODES:
        n.add("Bus", node, carrier="AC")
        n.add("Bus", f"{node} H2", carrier="H2")
    n.add("Bus", "EU gas", carrier="gas")
    n.add("Bus", "EU solid biomass", carrier="solid biomass")
    n.add("Bus", "EU waste", carrier="waste")
    n.add("Bus", "co2 stored", carrier="co2 stored")
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Bus", "EU coal", carrier="coal")
    n.add("Carrier", "coal")
    return n


def _routes_network() -> pypsa.Network:
    n = _base_network()
    costs, options, spatial = _costs(), _options(), _spatial()
    add_steel(n, costs, spatial, options, pd.Series([1.9, 0.5], index=NODES))
    add_cement(n, costs, spatial, options, pd.Series([2.1, 0.0], index=NODES))
    return n


def test_route_components_have_unique_names_and_demand_suffixes():
    n = _routes_network()

    check_unique_component_names(n)

    for node in NODES:
        for material in ("steel", "cement"):
            bus = f"{node} {material}"
            load = f"{bus} demand"
            assert n.buses.at[bus, "unit"] == "t"
            assert n.loads.at[load, "bus"] == bus
            assert n.loads.at[load, "carrier"] == material
    # the two-hour network carries the annual tonnes over its own hours
    assert n.loads.at["N0 steel demand", "p_set"] == pytest.approx(1.9e6 / 2)
    assert n.loads.at["N0 cement demand", "p_set"] == pytest.approx(2.1e6 / 2)
    # every emission bus is drained by a vent and a capture link
    for source in ("gas DRI emission", "BOF emission", "cement emission"):
        assert f"N0 {source}" in n.buses.index
        assert n.links.at[f"N0 {source} vent", "bus1"] == "co2 atmosphere"
        assert n.links.at[f"N0 {source} CC", "bus1"] == "co2 stored"


def test_route_links_convert_tonnes_at_catalogue_intensities():
    n = _routes_network()
    costs = _costs()

    bof = n.links.loc["N0 BOF"]
    assert bof.bus0 == "EU coal"
    assert bof.efficiency == pytest.approx(1 / 5.34)
    assert bof.efficiency2 == pytest.approx(0.3)
    dri = n.links.loc["N0 DRI"]
    assert dri.bus1 == "N0 hbi"
    assert dri.bus2 == "N0 DRI reduction"
    assert dri.marginal_cost == pytest.approx(97.7 * 1.59 / 1.03)
    kiln = n.links.loc["N0 clinker kiln"]
    assert kiln.bus0 == "N0 cement heat"
    assert kiln.bus1 == "N0 clinker"
    assert kiln.efficiency4 == pytest.approx((0.5071 + 0.3 * 0.0002) / 0.9444)
    finishing = n.links.loc["N0 cement production"]
    assert finishing.efficiency == pytest.approx(1 / 0.656)
    assert finishing.efficiency2 == pytest.approx(-0.1736 / 0.656)
    assert costs.at["cement finishing", "clinker-input"] == 0.656


def test_route_endogenised_sectors_follow_the_switch():
    assert route_endogenised_sectors({"enable": False, "subsectors": ["steel"]}) == []
    assert route_endogenised_sectors({"enable": True, "subsectors": ["cement"]}) == [
        "Cement"
    ]
    assert route_endogenised_sectors(
        {"enable": True, "subsectors": ["steel", "cement"]}
    ) == ["Integrated steelworks", "DRI + Electric arc", "Cement"]


def test_existing_industry_plants_become_dated_links(tmp_path):
    n = _routes_network()
    n.add("Bus", "EU methanol", carrier="methanol")
    n.add("Bus", "EU NH3", carrier="NH3")
    plants = pd.DataFrame(
        {
            "bus": ["N0", "N0", "N0", "N1", "N0", "N0", "N0"],
            "country": ["NL"] * 7,
            "carrier": [
                "BOF",
                "BOF",
                "gas DRI",
                "cement",
                "Haber-Bosch",
                "grey methanol",
                "cement",
            ],
            "p_set": [1.0e6, 0.5e6, 0.2e6, 1.5e6, 0.4e6, 0.1e6, 0.8e6],
            "build_year": [2012, float("nan"), 2018, 2005, 1998, 2001, 2012],
            "Out": [2030, 2030, 0, 0, 0, 0, 0],
        }
    )
    fn = tmp_path / "industry_plants.csv"
    plants.to_csv(fn, index=False)

    add_existing_industry(
        n,
        options=_options(),
        plant_fn=str(fn),
        grouping_years=[1995, 2000, 2005, 2010, 2015, 2020, 2025],
        baseyear=2025,
        costs=_costs(),
        spatial=_spatial(),
        industry_params={"MWh_NH3_per_tNH3": 5.166, "MWh_MeOH_per_tMeOH": 5.528},
    )

    check_unique_component_names(n)
    # the two blast furnaces share a grouping year (the NaN takes the mean, 2012)
    bof = n.links[n.links.carrier == "BOF"]
    existing = bof[~bof.p_nom_extendable]
    assert list(existing.index) == ["N0 BOF-2010-2030"]
    assert existing.p_nom.iloc[0] == pytest.approx(1.5e6 * 5.34 / 8760)
    assert existing.build_year.iloc[0] == 2010
    assert existing.lifetime.iloc[0] == 2030 - 2010  # the pledged phase-out
    dri = n.links.loc["N0 gas DRI-2015"]
    assert dri.carrier == "DRI" and dri.build_year == 2015
    # cement plants only where the network has a cement demand: N1 has none
    assert "N1 clinker kiln-2005" not in n.links.index
    assert "N1 cement production-2005" not in n.links.index
    # the clinker capacity sizes an existing kiln (heat) and its grinding (clinker)
    assert n.links.at["N0 clinker kiln-2010", "p_nom"] == pytest.approx(
        0.8e6 * 0.9444 / 8760
    )
    assert n.links.at["N0 cement production-2010", "p_nom"] == pytest.approx(
        0.8e6 / 8760
    )
    hb = n.links.loc["N0 Haber-Bosch-1995"]
    assert hb.bus1 == "EU NH3" and hb.p_nom == pytest.approx(
        0.4e6 * 5.166 / 0.25 / 8760
    )
    meoh = n.links.loc["N0 grey methanol-2000"]
    assert meoh.bus0 == "EU gas" and meoh.bus1 == "EU methanol"
