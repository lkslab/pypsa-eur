# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Waste configuration.

Waste streams (non-sequestered HVC, i.e. plastics, and municipal solid waste)
as their own sector, decoupled from the biomass and industry chapters
(port of PyPSA/pypsa-eur#1654).
"""

from pydantic import Field

from scripts.lib.validation.config._base import ConfigModel


class WasteConfig(ConfigModel):
    """Configuration for `waste` settings."""

    transport: bool = Field(
        False,
        description="Allow the transport of waste between model regions at the biomass transport costs; needs `sector.waste_spatial`.",
    )
    waste_to_energy: bool = Field(
        False,
        description="Expandable waste-to-energy CHPs that burn the waste potential for electricity and district heat.",
    )
    waste_to_energy_cc: bool = Field(
        False,
        description="Expandable waste-to-energy CHPs with carbon capture.",
    )
