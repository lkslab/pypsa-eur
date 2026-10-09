# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build industrial energy demand per model region.

Description
-------
This rule aggregates the energy demand of the industrial sectors per model region.
For each bus, the following carriers are considered:
- electricity
- coal
- coke
- solid biomass
- methane
- hydrogen
- low-temperature heat
- naphtha
- ammonia
- process emission
- process emission from feedstock

which can later be used as values for the industry load.
"""

import json
import logging

import holidays
import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

logger = logging.getLogger(__name__)

# FfE open data id 59: normed hourly electricity load profiles per NACE branch
# for Germany 2017 (eXtremOS project, CC-BY-4.0), by internal id. "Industry
# (total)", "Chemical Industry" and "Non-ferrous Metals" are not published.
FFE_PROFILE_NAMES = {
    1: "Iron & steel industry",
    4: "Non-metallic Minerals",
    5: "Transport Equipment",
    6: "Machinery",
    7: "Mining and Quarrying",
    8: "Food and Tobacco",
    9: "Paper, Pulp and Print",
    10: "Wood and Wood Products",
    11: "Construction",
    12: "Textile and Leather",
    13: "Non-specified (Industry)",
}
FFE_REFERENCE_YEAR = 2017

# JRC-IDEES sector -> FfE branch profile. Where FfE has no profile the nearest
# operating pattern is taken (PyPSA/pypsa-eur#1875): chemicals run continuously
# like paper mills (Ganz et al. 2021), non-ferrous metals like steel works,
# pharmaceuticals in batches with reduced weekends like food.
INDUSTRY_SECTOR_TO_PROFILE = {
    "Electric arc": "Iron & steel industry",
    "DRI + Electric arc": "Iron & steel industry",
    "Integrated steelworks": "Iron & steel industry",
    "HVC": "Paper, Pulp and Print",
    "HVC (mechanical recycling)": "Non-metallic Minerals",
    "HVC (chemical recycling)": "Non-metallic Minerals",
    "Ammonia": "Paper, Pulp and Print",
    "Chlorine": "Paper, Pulp and Print",
    "Methanol": "Paper, Pulp and Print",
    "Other chemicals": "Paper, Pulp and Print",
    "Pharmaceutical products etc.": "Food and Tobacco",
    "Cement": "Non-metallic Minerals",
    "Ceramics & other NMM": "Non-metallic Minerals",
    "Glass production": "Non-metallic Minerals",
    "Pulp production": "Paper, Pulp and Print",
    "Paper production": "Paper, Pulp and Print",
    "Printing and media reproduction": "Paper, Pulp and Print",
    "Food, beverages and tobacco": "Food and Tobacco",
    "Alumina production": "Iron & steel industry",
    "Aluminium - primary production": "Iron & steel industry",
    "Aluminium - secondary production": "Iron & steel industry",
    "Other non-ferrous metals": "Iron & steel industry",
    "Transport equipment": "Transport Equipment",
    "Machinery equipment": "Machinery",
    "Textiles and leather": "Textile and Leather",
    "Wood and wood products": "Wood and Wood Products",
    "Other industrial sectors": "Non-specified (Industry)",
}


def load_ffe_profiles(fn: str) -> pd.DataFrame:
    """The FfE branch profiles as an hourly 2017 frame, one column per branch."""
    with open(fn) as f:
        data = json.load(f)
    rows = {FFE_PROFILE_NAMES[r["internal_id"][0]]: r["values"] for r in data["data"]}
    index = pd.date_range(
        f"{FFE_REFERENCE_YEAR}-01-01", f"{FFE_REFERENCE_YEAR}-12-31 23:00", freq="h"
    )
    profiles = pd.DataFrame(rows, index=index)
    return profiles.div(profiles.sum())


def _weekday_hour_means(profile: pd.DataFrame) -> pd.DataFrame:
    return profile.groupby([profile.index.dayofweek, profile.index.hour]).mean()


def _at(means: pd.DataFrame, index: pd.DatetimeIndex, dayofweek=None) -> np.ndarray:
    """The weekday-hour mean of every timestamp, optionally for a fixed weekday."""
    dow = np.full(len(index), dayofweek) if dayofweek is not None else index.dayofweek
    keys = pd.MultiIndex.from_arrays([dow, index.hour])
    return means.reindex(keys).to_numpy()


def map_profiles_to_snapshots(
    profiles: pd.DataFrame, snapshots: pd.DatetimeIndex, country: str
) -> pd.DataFrame:
    """
    Carry the 2017 reference profiles onto the target snapshots.

    The reference year is shifted so weekdays line up, German holidays in the
    reference are replaced by the weekday-hour mean, the target country's own
    holidays get the Sunday shape, a leap day is filled with the weekday-hour
    mean, and every column is renormalised to sum to one over the snapshots.
    """
    means = _weekday_hour_means(profiles)
    ref = profiles.copy()

    de_holidays = holidays.country_holidays("DE", years=FFE_REFERENCE_YEAR)
    on_holiday = ref.index.normalize().isin(pd.to_datetime(list(de_holidays)))
    if on_holiday.any():
        ref.loc[on_holiday] = _at(means, ref.index[on_holiday])

    # the hourly values of the reference days shifted onto the target calendar
    # so that weekdays coincide
    target_year = snapshots[0].year
    # move the reference days by at most three days so their weekdays coincide
    # with the target year's calendar
    drift = (
        pd.Timestamp(year=FFE_REFERENCE_YEAR, month=1, day=1).dayofweek
        - pd.Timestamp(year=target_year, month=1, day=1).dayofweek
    ) % 7
    if drift > 3:
        drift -= 7
    shifted = ref.copy()
    shifted.index = shifted.index.map(lambda t: t.replace(year=target_year))
    shifted = shifted.shift(drift, freq="D") if drift else shifted
    hourly = pd.date_range(
        f"{target_year}-01-01", f"{target_year}-12-31 23:00", freq="h"
    )
    shifted = shifted.reindex(hourly)
    missing = shifted.isna().any(axis=1)
    if missing.any():
        shifted.loc[missing] = _at(means, hourly[missing])

    try:
        own_holidays = holidays.country_holidays(country, years=target_year)
        on_holiday = hourly.normalize().isin(pd.to_datetime(list(own_holidays)))
        if on_holiday.any():
            shifted.loc[on_holiday] = _at(means, hourly[on_holiday], dayofweek=6)
    except NotImplementedError:
        logger.warning(f"No holiday calendar for {country}; none imposed.")

    mapped = shifted.reindex(snapshots)
    missing = mapped.isna().any(axis=1)
    if missing.any():
        mapped.loc[missing] = _at(means, snapshots[missing])
    return mapped.div(mapped.sum())


def build_electricity_profiles(
    sector_electricity: pd.DataFrame,
    countries: pd.Series,
    ffe_profiles: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    One normalised hourly electricity profile per node (columns sum to one).

    Parameters
    ----------
    sector_electricity : pd.DataFrame
        Electricity demand per node (index) and JRC-IDEES sector (columns), any
        unit; only the sector shares per node matter.
    countries : pd.Series
        Country code per node, for the holiday calendar.
    """
    unmapped = sector_electricity.columns.difference(INDUSTRY_SECTOR_TO_PROFILE)
    if len(unmapped):
        raise ValueError(f"No FfE profile mapped for industry sectors {list(unmapped)}")
    # sector shares per node; a node without industry electricity gets a flat profile
    shares = sector_electricity.clip(lower=0)
    totals = shares.sum(axis=1)
    shares = shares.div(totals.where(totals > 0, 1.0), axis=0)
    branch_shares = shares.T.groupby(INDUSTRY_SECTOR_TO_PROFILE).sum().T

    profiles = pd.DataFrame(
        index=snapshots, columns=sector_electricity.index, dtype=float
    )
    for country, nodes in countries.groupby(countries).groups.items():
        mapped = map_profiles_to_snapshots(ffe_profiles, snapshots, country)
        profiles[nodes] = (
            mapped[branch_shares.columns].to_numpy()
            @ branch_shares.loc[nodes].T.to_numpy()
        )
    flat = totals <= 0
    if flat.any():
        profiles.loc[:, flat[flat].index] = 1.0 / len(snapshots)
    return profiles


sectors_copied_from_exogenous = [
    # also mainly has a heat demand, but can be electrified by methods unavailable in other processes.
    # Therefore, here exogenously electrified as long as individual sectors are not disaggregated.
    "HVC (chemical recycling)",
    "HVC (mechanical recycling)",
    "DRI + Electric arc",
]

# the JRC-IDEES sectors a route-endogenised subsector replaces (prepare_sector_network
# builds their material demand from the production file instead); secondary
# "Electric arc" steel stays an exogenous electricity demand
ROUTE_ENDOGENOUS_SECTORS = {
    "steel": ["Integrated steelworks", "DRI + Electric arc"],
    "cement": ["Cement"],
}


def route_endogenised_sectors(endogenous_sectors: dict) -> list[str]:
    """The JRC-IDEES sectors to drop from both demand blocks, [] when disabled."""
    if not endogenous_sectors.get("enable", False):
        return []
    return [
        sector
        for subsector in endogenous_sectors.get("subsectors", [])
        for sector in ROUTE_ENDOGENOUS_SECTORS.get(subsector, [])
    ]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_industrial_energy_demand_per_node",
            horizon=2030,
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # import exogenous ratios
    fn = snakemake.input.industry_sector_ratios
    sector_ratios_exogenous = pd.read_csv(fn, header=[0, 1], index_col=0)

    # import endogenous ratios
    fn = snakemake.input.industry_sector_ratios_endogenous
    sector_ratios_endogenous = pd.read_csv(fn, header=[0, 1], index_col=0)

    cols = sector_ratios_exogenous.columns[
        sector_ratios_exogenous.columns.get_level_values(1).isin(
            sectors_copied_from_exogenous
        )
    ]
    sector_ratios_endogenous = pd.concat(
        [sector_ratios_endogenous, sector_ratios_exogenous.loc[:, cols]], axis=1
    )

    # material demand per node and industry (Mton/a)
    fn = snakemake.input.industrial_production_per_node
    nodal_production = pd.read_csv(fn, index_col=0) / 1e3

    # energy demand today to get current electricity
    fn = snakemake.input.industrial_energy_demand_per_node_today
    nodal_today = pd.read_csv(fn, index_col=0)

    nodal_sector_ratios_exogenous = pd.concat(
        {node: sector_ratios_exogenous[node[:2]] for node in nodal_production.index},
        axis=1,
    )

    nodal_sector_ratios_endogenous = pd.concat(
        {node: sector_ratios_endogenous[node[:2]] for node in nodal_production.index},
        axis=1,
    )

    dropped = route_endogenised_sectors(snakemake.params.endogenous_sectors)
    if dropped:
        logger.info(
            f"Route-endogenised sectors left out of the energy demand: {dropped}"
        )
        nodal_production = nodal_production.drop(columns=dropped)

    nodal_production_stacked = nodal_production.stack()
    nodal_production_stacked.index.names = [None, None]

    # final energy consumption per node and industry (TWh/a)
    nodal_df_exogenous = (
        (nodal_sector_ratios_exogenous.multiply(nodal_production_stacked))
        .T.groupby(level=0)
        .sum()
    )

    nodal_df_endogenous = (
        (nodal_sector_ratios_endogenous.multiply(nodal_production_stacked))
        .T.groupby(level=0)
        .sum()
    )

    rename_sectors = {
        "elec": "electricity",
        "biomass": "solid biomass",
    }

    nodal_df_exogenous.rename(columns=rename_sectors, inplace=True)
    nodal_df_endogenous.rename(columns=rename_sectors, inplace=True)

    nodal_df_exogenous.rename(
        columns={
            "heat": "low-temperature heat",
        },
        inplace=True,
    )

    grouper = {
        "heat<100": "low-temperature heat",
        "heat": "low-temperature heat",
    }

    def partial_group(df, grouper):
        return pd.concat([df.groupby(grouper).sum(), df.drop(grouper)])

    nodal_df_endogenous = partial_group(nodal_df_endogenous.T, grouper).T

    # Process emissions (calcination / chemical-reaction CO2) and the naphtha and
    # methanol feedstocks are non-energy quantities. The endogenous ratios are derived
    # purely from fuel demand, so these carriers cannot be represented there and would
    # vanish when `industry_t: endogen` is enabled. They are independent of how process
    # heat is supplied, so carry them over from the exogenous block. Naphtha is copied
    # together with the feedstock emissions so the feedstock-emission / naphtha ratio in
    # prepare_sector_network stays self-consistent.
    for carrier in [
        "process emission",
        "process emission from feedstock",
        "naphtha",
        "methanol",
    ]:
        nodal_df_endogenous[carrier] = nodal_df_exogenous[carrier]

    # JRC-IDEES gives HVC a negative process emission in later horizons; a node
    # whose other sectors do not outweigh it would carry a negative emission load
    # that prepare_sector_network turns into an unservable CO2 withdrawal
    for block_name, block in [
        ("exogenous", nodal_df_exogenous),
        ("endogenous", nodal_df_endogenous),
    ]:
        negative = block.index[block["process emission"] < 0]
        if len(negative):
            logger.warning(
                f"Negative process emissions clipped to zero in the {block_name} block "
                f"at {len(negative)} node(s): {list(negative)}"
            )
            block.loc[negative, "process emission"] = 0.0

    nodal_df_exogenous["current electricity"] = nodal_today["electricity"]
    nodal_df_endogenous["current electricity"] = nodal_today["electricity"]

    nodal_df = pd.concat(
        [nodal_df_exogenous, nodal_df_endogenous],
        axis=1,
        keys=["exogenous", "endogenous"],
    )

    nodal_df.index.name = "TWh/a (MtCO2/a)"

    fn = snakemake.output.industrial_energy_demand_per_node
    nodal_df.to_csv(fn, float_format="%.2f")

    # hourly electricity profile per node from the sector mix of the exogenous
    # block, written always; industry.temporal_electricity_industry_load decides
    # in compose_network whether it is used
    sector_electricity = (
        nodal_sector_ratios_exogenous.loc["elec"].multiply(nodal_production_stacked)
    ).unstack()
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )
    profiles = build_electricity_profiles(
        sector_electricity=sector_electricity,
        countries=pop_layout.loc[sector_electricity.index, "ct"],
        ffe_profiles=load_ffe_profiles(snakemake.input.ffe_profiles),
        snapshots=snapshots,
    )
    profiles.to_csv(
        snakemake.output.industrial_electricity_profile, float_format="%.6e"
    )
