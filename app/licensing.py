#Imports
from dataclasses import dataclass
from datetime import date

@dataclass
class DataSource:
    """A class to hold attribution info for a single data source."""
    name: str
    attribution_template: str
    url: str
    notes: str = ""

#Central dataset for all data sources used
SOURCES = {
    "ons_boundaries": DataSource(
        name="ONS Open Geography",
        attribution_template="Source: Office for National Statistics licensed under the Open Government Licence v3.0. "
                             "Contains OS data © Crown copyright and database right {year}.",
        url="https://geoportal.statistics.gov.uk/",
        notes="Used for LSOA, Ward, and Local Authority (LAD) base polygons. LSOA geometries are the "
              "primary geographical unit for all data merging and calculations."
    ),
    "onspd": DataSource(
        name="ONS Postcode Directory (ONSPD)",
        attribution_template="Contains OS data © Crown copyright and database right {year}. "
                             "Contains Royal Mail data © Royal Mail copyright and database right {year}. "
                             "Source: Office for National Statistics licensed under the Open Government Licence v3.0.",
        url="https://geoportal.statistics.gov.uk/datasets/ons-postcode-directory/",
        notes="Used for the 'Find by Postcode' search functionality and to geocode the static locations of all "
              "GP practices and childcare providers."
    ),
    "police_crime": DataSource(
        name="Police UK Crime Data",
        attribution_template="Source: police.uk licensed under the Open Government Licence v3.0. "
                             "Contains data © Crown copyright and database right {year}.",
        url="https://data.police.uk/",
        notes="Used for crime-related indicators. Monthly LSOA-level data is aggregated annually to calculate a "
              "'crime rate per 1,000 people' (using ONS population). This rate is then ranked to create the Community Safety score."
    ),
    "imd": DataSource(
        name="Index of Multiple Deprivation (IMD)",
        attribution_template="Source: Ministry of Housing, Communities & Local Government licensed under the Open Government Licence v3.0. "
                             "Contains data © Crown copyright and database right {year}.",
        url="https://www.gov.uk/government/collections/english-indices-of-deprivation",
        notes="Used for deprivation scores. Provides the static 2019 LSOA-level deciles for "
              "Overall Deprivation, Income, Employment, and Health."
    ),
    "greenspace": DataSource(
        name="OS Open Greenspace",
        attribution_template="Contains OS data © Crown copyright and database right {year}. "
                             "Source: Ordnance Survey (via data.gov.uk) licensed under the Open Government Licence v3.0.",
        url="https://www.data.gov.uk/dataset/4c1fe120-a920-4f6d-bc41-8fd4586bd662/os-open-greenspace1",
        notes="Used to calculate the percentage of each LSOA's total area that is covered by a greenspace polygon. "
              "This percentage is then ranked to create the Greenspace score."
    ),
    "uk_air": DataSource(
        name="UK-AIR Annual Mean Maps",
        url="https://uk-air.defra.gov.uk/data/gis-mapping/",
        attribution_template="Contains Defra data © Crown copyright and database right {year}.",
        notes="Used to calculate Air Quality. Based on the annual mean concentration (in µg/m³) for "
              "Nitrogen Dioxide (NO₂) and fine particulate matter (PM₂.₅)."
    ),
    "ons_population": DataSource(
        name="ONS Mid-Year Population Estimates",
        url="https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates",
        attribution_template="Source: ONS © Crown copyright and database right {year}.",
        notes="Used to provide annual LSOA population estimates. This data is displayed as a core metric and is "
              "used as the denominator for calculating the 'crime rate per 1,000 people'."
    ),
    "dfe_schools": DataSource(
        name="DfE Find and Compare School Performance",
        url="https://www.compare-school-performance.service.gov.uk/",
        attribution_template="Source: Department for Education © Crown copyright {year}.",
        notes="Used to calculate School Performance. The Primary score is a ranked average of 'KS2 pass rate' and "
              "'KS2 average scaled score' from the 3 nearest schools. The Secondary score is a ranked average of "
              "'Progress 8' and 'Attainment 8' scores from the 3 nearest schools."
    ),
    "nhs_digital": DataSource(
        name="NHS Digital (epraccur)",
        url="https://digital.nhs.uk/services/organisation-data-service/data-search-and-export/csv-downloads/gp-and-gp-practice-related-data",
        attribution_template="Contains NHS Digital data © {year} licensed under the Open Government Licence v3.0.",
        notes="Used to find the 3 nearest active GP practices to each LSOA. This file provides the master list of "
              "practices and their postcodes, which are then geocoded to calculate distances."
    ),
    "gp_survey": DataSource(
        name="GP Patient Survey",
        url="https://gp-patient.co.uk/",
        attribution_template="Contains GP Patient Survey data © {year} licensed under the Open Government Licence v3.0.",
        notes="Used to calculate the GP Patient Satisfaction Score. The annual satisfaction score for the 3 nearest GPs "
              "is averaged, and this average is then ranked. The score is based on the definitive 'Overall Experience' "
              "summary column for each year's data file (e.g., 'overallexp.pcteval' or 'Q28_12pct')."
    ),
    "ofsted_childcare": DataSource(
        name="Ofsted Childcare Providers and Inspections",
        url="https://www.gov.uk/government/statistical-data-sets/childcare-providers-and-inspections-management-information",
        attribution_template="Contains Ofsted data © Crown copyright and database right {year} licensed under the Open Government Licence v3.0.",
        notes="Used to find the 3 nearest childcare providers. The Childcare score is a ranked average of three metrics: "
              "the average distance to these providers, their average Ofsted quality rating (1-4), and the sum of their "
              "total registered places."
    ),
    "ons_house_prices": DataSource(
        name="House price statistics for small areas",
        url="https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/housepricestatisticsforsmallareas", # Link to the main dataset page
        attribution_template="Contains HM Land Registry data © Crown copyright and database right {year}. Contains Ordnance Survey data © Crown copyright and database right {year}. Contains Royal Mail data © Royal Mail copyright and database right {year}. Contains National Statistics data © Crown copyright and database right {year}.",
        notes="Used to show median house prices over time at LSOA and Ward level. Based on data derived from HM Land Registry Price Paid data."
    )
}

def get_source(id: str) -> DataSource:
    """Helper function to retrieve a data source by its key."""
    return SOURCES.get(id)


def generate_attribution_markdown() -> str:
    """Generates a markdown string listing attributions for all data sources."""
    current_year = date.today().year
    markdown_lines = []

    #Sort sources by name
    sorted_sources = sorted(SOURCES.items(), key=lambda item: item[1].name)

    for _, source in sorted_sources:
        formatted_attribution = source.attribution_template.format(year=current_year)
        markdown_lines.append(f"- **[{source.name}]({source.url}):** {formatted_attribution}")
    return "\n".join(markdown_lines)