# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Locate today's steel, cement, ammonia and methanol plants on the model regions.

Description
-----------
Builds one table of existing industry production units, ``industry_plants.csv``,
with the columns ``bus`` (model region), ``country``, ``carrier`` (the route:
``BOF``, ``gas DRI``, ``Haber-Bosch``, ``grey methanol``, ``cement``), ``p_set``
(production in t/a; for ``cement`` the plant's clinker capacity in t/a, since a
grinding-only site has no kiln and is left out), ``build_year`` and ``Out``
(pledged decommissioning year, 0 when none). ``add_existing_baseyear`` turns the rows into brownfield links when
``sector.endogenous_sectors.enable`` is set.

Sources: the Fraunhofer ISI industry site database (Neuwirth et al.) for steel,
ammonia and methanol in the EU27 plus GB and NO, the GEM Global Cement and
Concrete Tracker for cement plants, and ``data/ammonia_plants.csv`` for ammonia
plants in countries the ISI database does not cover. Port of PyPSA/pypsa-eur#1719.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pycountry

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

COLUMNS = ["bus", "country", "carrier", "p_set", "build_year", "Out"]

# Eurostat country codes the ISI database uses
ISI_COUNTRY_CODES = {"UK": "GB", "EL": "GR"}

ISI_ROUTES = {
    "Blast furnace": "BOF",
    "Direct reduction NG": "gas DRI",
    "Ammonia SMR": "Haber-Bosch",
    "Methanol SMR": "grey methanol",
}


def country_to_code(name: str) -> str | None:
    """ISO alpha-2 code of a country name, Kosovo included; None when unknown."""
    if name == "Kosovo":
        return "XK"
    try:
        return pycountry.countries.lookup(name).alpha_2
    except LookupError:
        return None


def _assign_regions(df: pd.DataFrame, regions: gpd.GeoDataFrame) -> pd.DataFrame:
    """Map plants to the nearest onshore region; drop plants without coordinates."""
    df = df.dropna(subset=["Longitude", "Latitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin_nearest(gdf, regions.to_crs("EPSG:4326"), how="left")
    joined = joined.rename(columns={"name": "bus"})
    missing = joined["bus"].isna().sum()
    if missing:
        logger.warning(f"{missing} plants could not be mapped to a region, dropped.")
    return pd.DataFrame(joined.drop(columns="geometry")).dropna(subset=["bus"])


def prepare_gem_cement_plants(
    fn: str, regions: gpd.GeoDataFrame, countries: list[str]
) -> pd.DataFrame:
    """
    Operating integrated cement plants from the GEM Global Cement and Concrete
    Tracker, with their clinker capacity in t/a. Grinding-only sites have no kiln
    (NL's three sites since ENCI Maastricht closed in 2019) and are left out.
    """
    df = pd.read_excel(fn, sheet_name="Plant Data", na_values=["N/A", "unknown", ">0"])
    df["country"] = df["Country/Area"].map(country_to_code)
    df = df[
        df["country"].isin(countries)
        & (df["Operating status"] == "operating")
        & (df["Plant type"] == "integrated")
    ]

    latlon = df["Coordinates"].str.split(",", expand=True)
    df = df.assign(
        Latitude=pd.to_numeric(latlon[0].str.strip(), errors="coerce"),
        Longitude=pd.to_numeric(latlon[1].str.strip(), errors="coerce"),
        build_year=pd.to_numeric(
            df["Start date"].astype(str).str.split("-").str[0], errors="coerce"
        ),
        p_set=df["Clinker Capacity (millions metric tonnes per annum)"].fillna(0) * 1e6,
        carrier="cement",
        Out=0,
    )
    df = df[df["p_set"] > 0]
    df = _assign_regions(df, regions)
    return df[COLUMNS].reset_index(drop=True)


def prepare_isi_plants(
    fn: str, regions: gpd.GeoDataFrame, countries: list[str]
) -> pd.DataFrame:
    """Steel, ammonia and methanol units from the Fraunhofer ISI site database."""
    df = pd.read_excel(fn, sheet_name="Database", index_col=1)
    df["country"] = df["Country"].replace(ISI_COUNTRY_CODES)
    df = df[df["country"].isin(countries)]
    df = df[df["Process status qup"].isin(ISI_ROUTES)]
    df = df.assign(
        carrier=df["Process status qup"].map(ISI_ROUTES),
        build_year=pd.to_numeric(
            df["Year of last modernisation"].replace("x", np.nan), errors="coerce"
        ).fillna(pd.to_numeric(df["Last Relining"], errors="coerce")),
        p_set=df["Production in tons (calibrated)"],
        Out=pd.to_numeric(df["Out"], errors="coerce").fillna(0),
    )
    df = _assign_regions(df, regions)
    return df[COLUMNS].reset_index(drop=True)


def prepare_ammonia_plants(
    fn: str,
    regions: gpd.GeoDataFrame,
    countries: list[str],
    isi_plants: pd.DataFrame,
) -> pd.DataFrame:
    """Ammonia plants for the countries the ISI database does not cover."""
    df = pd.read_csv(fn, index_col=0)
    df = _assign_regions(df, regions)
    df["country"] = df["bus"].str[:2]
    covered = set(isi_plants.loc[isi_plants.carrier == "Haber-Bosch", "country"])
    df = df[df["country"].isin(countries) & ~df["country"].isin(covered)]

    # as in build_industrial_distribution_key: a plant without a figure gets half
    # the smallest plant of its country
    min_prod = df.groupby("country")["Ammonia [kt/a]"].transform("min")
    df["Ammonia [kt/a]"] = df["Ammonia [kt/a]"].fillna(0.5 * min_prod)
    df = df.dropna(subset=["Ammonia [kt/a]"])

    avg_year = isi_plants.loc[isi_plants.carrier == "Haber-Bosch", "build_year"].mean()
    df = df.assign(
        carrier="Haber-Bosch",
        p_set=df["Ammonia [kt/a]"] * 1e3,
        build_year=avg_year,
        Out=0,
    )
    return df[COLUMNS].reset_index(drop=True)


def build_industry_plants(
    regions: gpd.GeoDataFrame,
    countries: list[str],
    gem_fn: str,
    isi_fn: str,
    ammonia_fn: str,
) -> pd.DataFrame:
    cement = prepare_gem_cement_plants(gem_fn, regions, countries)
    isi = prepare_isi_plants(isi_fn, regions, countries)
    ammonia = prepare_ammonia_plants(ammonia_fn, regions, countries, isi)
    plants = pd.concat([cement, isi, ammonia], ignore_index=True)
    logger.info(
        "Industry plants per route (t/a): "
        + ", ".join(
            f"{c}: {v:.3g}" for c, v in plants.groupby("carrier")["p_set"].sum().items()
        )
    )
    return plants


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_industry_plants")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    regions = gpd.read_file(snakemake.input.onshore_regions).set_index("name")

    plants = build_industry_plants(
        regions=regions,
        countries=snakemake.params.countries,
        gem_fn=snakemake.input.gem_gcct,
        isi_fn=snakemake.input.isi_database,
        ammonia_fn=snakemake.input.ammonia,
    )
    plants.to_csv(snakemake.output.industry_plants, index=False)
