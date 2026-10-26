# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>>
#
# SPDX-License-Identifier: MIT
"""
Build spatial distribution of industries from Hotmaps database or the EU ETS sites.

Description
-------

With ``industry: distribution_key_source: hotmaps`` (default), this rule uses the `Hotmaps database <https://gitlab.com/hotmaps/industrial_sites/industrial_sites_Industrial_Database>`. After removing entries without valid locations, it assigns each industrial site to a bus region based on its location.
Then, it calculates the nodal distribution key for each sector based on the emissions of the industrial sites in each region. This leads to a distribution key of 1 if there is only one bus per country and <1 if there are multiple buses per country. The sum over buses of one country is 1.

With ``ets_sites``, the emissions are those of ``industrial_sites.csv`` from
``build_industrial_sites`` instead (EU ETS verified emissions of the latest
complete year, Great Britain from the NAEI), with its sectors mapped onto the
subcategories below (see ``ETS_SITE_SUBSECTORS``). Each site goes to a region of
its own country: the one it lies in, else the nearest within 10 km. Oil and gas
extraction is left out, as it is not industry demand in the model. A country
whose sites name no glassworks (Great Britain: the NAEI has one sector for all
minerals) spreads glass like its other mineral sites. In both cases the GEM
trackers set cement and the steel routes, the ammonia plants set ammonia, the
non-EU refineries supplement fills refineries where there are no sites, and
countries without sites (Switzerland, the Western Balkans) fall back to
population.

The following subcategories of industry are considered:
- Iron and steel
- Cement
- Refineries
- Paper and printing
- Chemical industry
- Glass
- Non-ferrous metals
- Non-metallic mineral products
- Other non-classified
Furthermore, the population distribution is added
- Population
"""

import logging
import uuid
from itertools import product

import country_converter as coco
import geopandas as gpd
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
cc = coco.CountryConverter()

SUBSECTORS = [
    "Iron and steel",
    "Cement",
    "Refineries",
    "Paper and printing",
    "Chemical industry",
    "Glass",
    "Non-ferrous metals",
    "Non-metallic mineral products",
    "Other non-classified",
]
# sectors of industrial_sites.csv (build_industrial_sites) onto the Hotmaps
# subsectors; minerals are split into glass and the rest by activity and NACE
ETS_SITE_SUBSECTORS = {
    "Iron and steel": "Iron and steel",
    "Cement and lime": "Cement",
    "Refining": "Refineries",
    "Pulp and paper": "Paper and printing",
    "Chemicals": "Chemical industry",
    "Glass, ceramics and other minerals": "Non-metallic mineral products",
    "Non-ferrous metals": "Non-ferrous metals",
    "Food and drink": "Other non-classified",
    "Other industry": "Other non-classified",
}
# a site further than this from its country's regions is left out, in metres
MAX_DISTANCE_M = 10_000.0


def locate_missing_industrial_sites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Locate industrial sites without valid locations based on city and
    countries.

    Should only be used if the model's spatial resolution is coarser
    than individual cities.
    """
    try:
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim
    except ImportError:
        raise ModuleNotFoundError(
            "Optional dependency 'geopy' not found."
            "Install via 'pixi add geopy'"
            "or set 'industry: hotmaps_locate_missing: false'."
        )

    locator = Nominatim(user_agent=str(uuid.uuid4()))
    geocode = RateLimiter(locator.geocode, min_delay_seconds=2)

    def locate_missing(s):
        if pd.isna(s.City) or s.City == "CONFIDENTIAL":
            return None

        loc = geocode([s.City, s.Country], geometry="wkt")
        if loc is not None:
            logger.debug(f"Found:\t{loc}\nFor:\t{s['City']}, {s['Country']}\n")
            return f"POINT({loc.longitude} {loc.latitude})"
        else:
            return None

    missing = df.index[df.geom.isna()]
    df.loc[missing, "coordinates"] = df.loc[missing].apply(locate_missing, axis=1)

    # report stats
    num_still_missing = df.coordinates.isna().sum()
    num_found = len(missing) - num_still_missing
    share_missing = len(missing) / len(df) * 100
    share_still_missing = num_still_missing / len(df) * 100
    logger.warning(
        f"Found {num_found} missing locations. \nShare of missing locations reduced from {share_missing:.2f}% to {share_still_missing:.2f}%."
    )

    return df


def prepare_hotmaps_database(fn: str, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load hotmaps database of industrial sites and map onto bus regions.
    """
    df = pd.read_csv(fn, sep=";", index_col=0)

    df[["srid", "coordinates"]] = df.geom.str.split(";", expand=True)

    if snakemake.params.hotmaps_locate_missing:
        df = locate_missing_industrial_sites(df)

    # remove those sites without valid locations
    df.drop(df.index[df.coordinates.isna()], inplace=True)

    df["coordinates"] = gpd.GeoSeries.from_wkt(df["coordinates"])

    gdf = gpd.GeoDataFrame(df, geometry="coordinates", crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    # the .sjoin can lead to duplicates if a geom is in two overlapping regions
    if gdf.index.duplicated().any():
        # get all duplicated entries
        duplicated_i = gdf.index[gdf.index.duplicated()]
        # convert from raw data country name to iso-2-code
        code = cc.convert(gdf.loc[duplicated_i, "Country"], to="iso2")  # noqa: F841
        # screen out malformed country allocation
        gdf_filtered = gdf.loc[duplicated_i].query("country == @code")
        # concat not duplicated and filtered gdf
        gdf = pd.concat([gdf.drop(duplicated_i), gdf_filtered])

    gdf["emissions"] = gdf["Emissions_ETS_2014"].fillna(gdf["Emissions_EPRTR_2014"])

    return gdf


def is_glassworks(activity: pd.Series, nace: pd.Series) -> pd.Series:
    """Whether a minerals site makes glass, by its ETS activity or NACE group."""
    by_activity = activity.fillna("").str.contains("glass", case=False)
    by_nace = nace.astype(str).str.strip().str.startswith("23.1")
    return by_activity | by_nace


def prepare_ets_sites(fn: str, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load the industrial sites of build_industrial_sites with their Hotmaps
    subsector and map them onto the bus regions of their own country.
    """
    df = pd.read_csv(fn, dtype={"nace": str})

    df["Subsector"] = df["sector"].map(ETS_SITE_SUBSECTORS)
    minerals = df["sector"] == "Glass, ceramics and other minerals"
    glass = minerals & is_glassworks(df["activity"], df["nace"])
    df.loc[glass, "Subsector"] = "Glass"
    df = df[df["Subsector"].notna()]

    # a country whose sites name no glassworks spreads glass like its minerals
    with_glass = df.loc[df["Subsector"] == "Glass", "country"].unique()
    stand_in = df[
        (df["Subsector"] == "Non-metallic mineral products")
        & ~df["country"].isin(with_glass)
    ]
    df = pd.concat([df, stand_in.assign(Subsector="Glass")], ignore_index=True)

    gdf = gpd.GeoDataFrame(
        df.rename(columns={"co2_t": "emissions"}),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(3035)
    buses = gpd.GeoDataFrame(
        {"bus": regions.index}, geometry=regions.geometry.values, crs=regions.crs
    ).to_crs(3035)
    buses["country"] = buses["bus"].str[:2]

    located = []
    for country, sites_ct in gdf.groupby("country"):
        buses_ct = buses.loc[buses["country"] == country, ["bus", "geometry"]]
        if buses_ct.empty:
            continue
        joined = gpd.sjoin_nearest(
            sites_ct, buses_ct, how="inner", max_distance=MAX_DISTANCE_M
        )
        # a site on the border of two regions is matched to both
        located.append(joined[~joined.index.duplicated()])

    gdf = pd.concat(located) if located else gdf.iloc[:0].assign(bus=None)

    dropped = df["country"].isin(buses["country"]).sum() - len(gdf)
    if dropped:
        logger.info(
            f"Left out {dropped} industrial sites further than "
            f"{MAX_DISTANCE_M / 1e3:g} km from their country's regions."
        )

    return gdf


def prepare_gem_steel_database(fn: str, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load GEM database of steel plants and map onto bus regions.
    """

    df1 = pd.read_excel(
        fn,
        sheet_name="Plant data",
        na_values=["N/A", "unknown", ">0"],
    )

    df2 = pd.read_excel(
        fn,
        sheet_name="Plant capacities and status",
        na_values=["N/A", "unknown", ">0"],
    )

    df = df1.merge(df2, on="Plant ID", suffixes=(None, " 2")).query(
        "Region == 'Europe'"
    )

    df["Retired date"] = pd.to_numeric(
        df["Retired date"].combine_first(df["Idled date"]), errors="coerce"
    )
    df["Start date"] = pd.to_numeric(
        df["Start date"].str.split("-").str[0], errors="coerce"
    )

    latlon = (
        df["Coordinates"]
        .str.split(", ", expand=True)
        .rename(columns={0: "lat", 1: "lon"})
    )
    geometry = gpd.points_from_xy(latlon["lon"], latlon["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    return gdf


def prepare_gem_cement_database(fn: str, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load GEM database of cement plants and map onto bus regions.
    """

    df = pd.read_excel(
        fn,
        sheet_name="Plant Data",
        na_values=["N/A", "n/a", "unknown", ">0"],
    )

    df["Start date"] = pd.to_numeric(df["Start date"], errors="coerce")

    df = df.loc[df.Coordinates.str.contains(", ") & df.Coordinates.notna()]
    latlon = (
        df["Coordinates"]
        .str.split(", ", expand=True)
        .rename(columns={0: "lat", 1: "lon"})
    )
    geometry = gpd.points_from_xy(latlon["lon"], latlon["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    return gdf


def prepare_ammonia_database(fn: str, regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load ammonia database of plants and map onto bus regions.
    """
    df = pd.read_csv(fn, index_col=0)

    geometry = gpd.points_from_xy(df.Longitude, df.Latitude)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    return gdf


def prepare_refineries_supplement(
    fn: str, regions: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Load supplementary refineries from non-EU-(NO-CH) and map onto bus regions.
    """

    df = pd.read_csv(fn, index_col=0)

    geometry = gpd.points_from_xy(df.Longitude, df.Latitude)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    return gdf


def build_nodal_distribution_key(
    sites: gpd.GeoDataFrame,
    steel: gpd.GeoDataFrame,
    ammonia: gpd.GeoDataFrame,
    cement: gpd.GeoDataFrame,
    refineries: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    countries: list[str],
) -> pd.DataFrame:
    """
    Build nodal distribution keys for each sector.

    `sites` are the Hotmaps sites or the EU ETS sites, with their
    ``Subsector``, ``bus``, ``country`` and ``emissions``.
    """
    sectors = list(sites.Subsector.unique())
    # a selection without sites of a subsector still gets its column
    sectors += [s for s in SUBSECTORS if s not in sectors]

    keys = pd.DataFrame(index=regions.index, columns=sectors, dtype=float)

    pop = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    pop["country"] = pop.index.str[:2]
    ct_total = pop.total.groupby(pop["country"]).sum()
    keys["population"] = pop.total / pop.country.map(ct_total)

    for sector, country in product(sectors, countries):
        regions_ct = regions.index[regions.index.str.contains(country)]

        facilities = sites.query("country == @country and Subsector == @sector")

        if not facilities.empty:
            emissions = facilities["emissions"]
            if emissions.sum() == 0:
                key = pd.Series(1 / len(facilities), facilities.index)
            else:
                # assume 20% quantile for missing values
                emissions = emissions.fillna(emissions.quantile(0.2))
                key = emissions / emissions.sum()
            key = key.groupby(facilities.bus).sum().reindex(regions_ct, fill_value=0.0)
        elif sector == "Refineries" and country in refineries.country.unique():
            facilities = refineries.query("country == @country")
            production = facilities["Capacity [bbl/day]"]
            if production.sum() == 0:
                key = pd.Series(1 / len(facilities), facilities.index)
            else:
                key = production / production.sum()
            key = key.groupby(facilities.bus).sum().reindex(regions_ct, fill_value=0.0)
        else:
            key = keys.loc[regions_ct, "population"]

        keys.loc[regions_ct, sector] = key

    # add cement distribution from GEM
    for country in countries:
        regions_ct = regions.index[regions.index.str.contains(country)]

        status_list = ["operating", "operating pre-retirement", "construction"]
        facilities = cement.query(
            "country == @country and `Operating status` in @status_list"
        )

        clinker_col = "Clinker Capacity (millions metric tonnes per annum)"
        cement_col = "Cement Capacity (millions metric tonnes per annum)"
        clinker_cement_ratio = 0.737  # https://lowcarboneconomy.cembureau.eu/5-parallel-routes/resource-efficiency/clinker-substitution

        # clinker capacity
        capacity_from_clinker = facilities[clinker_col]

        # approximated clinker capacity from cement capacity for integrated plants
        capacity_from_integrated_cement = facilities.query(
            "`Plant type` == 'integrated'"
        )[cement_col].mul(clinker_cement_ratio)

        # for grinding plants, use 10% of cement capacity as proxy for energy use
        capacity_from_grinding_cement = facilities.query("`Plant type` == 'grinding'")[
            cement_col
        ].mul(0.1 * clinker_cement_ratio)

        # assume 1% quantile for missing values
        missing_i = facilities.loc[
            facilities[clinker_col].isna() & facilities[cement_col].isna()
        ].index
        capacity_from_Q10 = pd.Series(
            cement[cement_col].quantile(0.01) * clinker_cement_ratio, index=missing_i
        )

        capacities = (
            capacity_from_clinker.combine_first(capacity_from_integrated_cement)
            .combine_first(capacity_from_grinding_cement)
            .combine_first(capacity_from_Q10)
        )

        if not capacities.empty:
            if capacities.sum() == 0:
                key = pd.Series(1 / len(capacities), capacities.index)
            else:
                key = capacities / capacities.sum()
            buses = facilities.loc[capacities.index, "bus"]
            key = key.groupby(buses).sum().reindex(regions_ct, fill_value=0.0)
        else:
            key = keys.loc[regions_ct, "population"]

        keys.loc[regions_ct, "Cement"] = key

    # add specific steel subsectors from GEM
    steel_processes = ["EAF", "DRI + EAF", "Integrated steelworks"]
    for process, country in product(steel_processes, countries):
        regions_ct = regions.index[regions.index.str.contains(country)]

        facilities = steel.query("country == @country")

        if process == "EAF":
            status_list = [
                "construction",
                "operating",
                "operating pre-retirement",
                "retired",
            ]
            capacities = facilities.loc[
                facilities["Status"].isin(status_list)
                & (
                    facilities["Retired date"].isna()
                    | facilities["Retired date"].gt(2025)
                ),
                "Nominal EAF steel capacity (ttpa)",
            ].dropna()
        elif process == "DRI + EAF":
            status_list = [
                "construction",
                "operating",
                "operating pre-retirement",
                "retired",
                "announced",
            ]
            sel = [
                "Nominal BOF steel capacity (ttpa)",
                "Nominal OHF steel capacity (ttpa)",
                "Nominal iron capacity (ttpa)",
            ]
            status_filter = facilities["Status"].isin(status_list)
            retirement_filter = facilities["Retired date"].isna() | facilities[
                "Retired date"
            ].gt(2030)
            start_filter = (
                facilities["Start date"].isna() & ~facilities["Status"].eq("announced")
            ) | facilities["Start date"].le(2030)
            capacities = (
                facilities.loc[status_filter & retirement_filter & start_filter, sel]
                .sum(axis=1)
                .dropna()
            )
        elif process == "Integrated steelworks":
            status_list = [
                "construction",
                "operating",
                "operating pre-retirement",
                "retired",
            ]
            sel = [
                "Nominal BOF steel capacity (ttpa)",
                "Nominal OHF steel capacity (ttpa)",
            ]
            capacities = (
                facilities.loc[
                    facilities["Status"].isin(status_list)
                    & (
                        facilities["Retired date"].isna()
                        | facilities["Retired date"].gt(2025)
                    ),
                    sel,
                ]
                .sum(axis=1)
                .dropna()
            )
        else:
            raise ValueError(f"Unknown process {process}")

        if not capacities.empty:
            if capacities.sum() == 0:
                key = pd.Series(1 / len(capacities), capacities.index)
            else:
                key = capacities / capacities.sum()
            buses = facilities.loc[capacities.index, "bus"]
            key = key.groupby(buses).sum().reindex(regions_ct, fill_value=0.0)
        else:
            key = keys.loc[regions_ct, "population"]

        keys.loc[regions_ct, process] = key

    # add ammonia
    for country in countries:
        regions_ct = regions.index[regions.index.str.contains(country)]

        facilities = ammonia.query("country == @country")

        if not facilities.empty:
            production = facilities["Ammonia [kt/a]"]
            if production.sum() == 0:
                key = pd.Series(1 / len(facilities), facilities.index)
            else:
                # assume 50% of the minimum production for missing values
                production = production.fillna(0.5 * facilities["Ammonia [kt/a]"].min())
                key = production / production.sum()
            key = key.groupby(facilities.bus).sum().reindex(regions_ct, fill_value=0.0)
        else:
            key = 0.0

        keys.loc[regions_ct, "Ammonia"] = key

    return keys


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_industrial_distribution_key")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    countries = snakemake.params.countries

    regions = gpd.read_file(snakemake.input.onshore_regions).set_index("name")

    if snakemake.params.distribution_key_source == "ets_sites":
        sites = prepare_ets_sites(snakemake.input.industrial_sites, regions)
    else:
        sites = prepare_hotmaps_database(snakemake.input.hotmaps, regions)

    steel = prepare_gem_steel_database(snakemake.input.gem_gspt, regions)

    ammonia = prepare_ammonia_database(snakemake.input.ammonia, regions)

    cement = prepare_gem_cement_database(snakemake.input.gem_gcpt, regions)

    refineries = prepare_refineries_supplement(
        snakemake.input.refineries_supplement, regions
    )

    keys = build_nodal_distribution_key(
        sites, steel, ammonia, cement, refineries, regions, countries
    )

    keys.to_csv(snakemake.output.industrial_distribution_key)
