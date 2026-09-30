"""
Jennette's Pier NOAA Storm Events Extractor
-------------------------------------------
Downloads NOAA/NCEI Storm Events data for 2024-2026,
finds reported storm-event locations within 50 statute miles
of Jennette's Pier, and creates:

1. Jennettes_Pier_NOAA_Weather_Events.xlsx
2. Jennettes_Pier_Weather_Data.json
3. Jennettes_Pier_Weather_Data.js

The JS file can be used directly by the interactive dashboard,
which avoids browser file-access/CORS problems when testing locally.

Requirements:
    pip install pandas requests openpyxl
"""

import io
import json
import math
import re
from pathlib import Path

import pandas as pd
import requests
import numpy as np


# ============================================================
# PROJECT SETTINGS
# ============================================================

PIER_NAME = "Jennette's Pier"
PIER_LAT = 35.9100
PIER_LON = -75.5917

RADIUS_MILES = 50.0
START_DATE = pd.Timestamp("2024-10-01")

OUTPUT_DIR = Path.cwd()

EXCEL_FILE = OUTPUT_DIR / "Jennettes_Pier_NOAA_Weather_Events.xlsx"
JSON_FILE = OUTPUT_DIR / "Jennettes_Pier_Weather_Data.json"
JS_FILE = OUTPUT_DIR / "Jennettes_Pier_Weather_Data.js"

NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"

# Years needed for the project.
YEARS = [2024, 2025, 2026]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_columns(df):
    """
    Normalize NOAA column names so the script works with
    uppercase/lowercase variations.
    """
    df = df.copy()

    new_columns = []

    for col in df.columns:
        col = str(col).strip().lower()
        col = col.replace(" ", "_")
        col = col.replace("-", "_")
        new_columns.append(col)

    df.columns = new_columns

    return df


def find_column(df, possible_names, required=False):
    """
    Find the first matching column from a list of possible names.
    """
    for name in possible_names:
        if name.lower() in df.columns:
            return name.lower()

    if required:
        raise KeyError(
            f"Could not find required column. "
            f"Expected one of: {possible_names}\n"
            f"Columns found:\n{list(df.columns)}"
        )

    return None


def download_csv_gz(url):
    """
    Download a compressed NOAA CSV and return a pandas DataFrame.
    """
    print(f"\nDownloading:")
    print(url)

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB")

    return pd.read_csv(
        io.BytesIO(response.content),
        compression="gzip",
        low_memory=False
    )


def get_noaa_file(year, file_type):
    """
    Find the newest NOAA file for a particular year/type.

    file_type should be:
        details
        locations

    NOAA filenames change revision dates, so we inspect the
    directory listing instead of relying on one hard-coded date.
    """
    prefix = f"StormEvents_{file_type}-ftp_v1.0_d{year}_c"
    suffix = ".csv.gz"

    print(f"\nLooking for NOAA {file_type} file for {year}...")

    response = requests.get(NOAA_BASE + "/", timeout=60)
    response.raise_for_status()

    # Find filenames in the directory listing.
    pattern = re.compile(
        rf'href=["\']?({re.escape(prefix)}\d+{re.escape(suffix)})["\']?',
        re.IGNORECASE
    )

    matches = pattern.findall(response.text)

    if not matches:
        # Some directory listings don't use href quotes exactly as expected.
        pattern = re.compile(
            rf'({re.escape(prefix)}\d+{re.escape(suffix)})',
            re.IGNORECASE
        )
        matches = pattern.findall(response.text)

    if not matches:
        raise FileNotFoundError(
            f"\nCould not find NOAA {file_type} file for {year}.\n"
            f"NOAA directory checked:\n{NOAA_BASE}\n"
        )

    # Pick the newest revision.
    filename = sorted(set(matches))[-1]

    print(f"Using: {filename}")

    return f"{NOAA_BASE}/{filename}"


def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance in statute miles.

    This version works with either individual numbers OR pandas
    Series/NumPy arrays. NOAA provides many locations at once,
    so the calculation must be vectorized.
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    # Protect against tiny floating-point values outside [0, 1].
    a = np.clip(a, 0, 1)

    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_miles = 3958.7613

    return earth_radius_miles * c


def clean_value(value):
    """
    Convert pandas/NumPy values into JSON-safe values.
    """
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


# ============================================================
# LOAD ONE YEAR
# ============================================================

def process_year(year):
    """
    Download and process one NOAA Storm Events year.

    Returns one row per NOAA EVENT_ID whose reported location
    is within the project radius.
    """

    print("\n" + "=" * 70)
    print(f"PROCESSING NOAA STORM EVENTS {year}")
    print("=" * 70)

    details_url = get_noaa_file(year, "details")
    locations_url = get_noaa_file(year, "locations")

    details = download_csv_gz(details_url)
    locations = download_csv_gz(locations_url)

    # Normalize column names.
    details = normalize_columns(details)
    locations = normalize_columns(locations)

    print("\nDetails columns:")
    print(list(details.columns))

    print("\nLocation columns:")
    print(list(locations.columns))

    # --------------------------------------------------------
    # DETAILS FILE
    # --------------------------------------------------------

    details_event_id = find_column(
        details,
        ["event_id"],
        required=True
    )

    details_event_type = find_column(
        details,
        ["event_type"],
        required=False
    )

    details_begin = find_column(
        details,
        ["begin_date_time", "begin_datetime"],
        required=False
    )

    details_end = find_column(
        details,
        ["end_date_time", "end_datetime"],
        required=False
    )

    details_state = find_column(
        details,
        ["state"],
        required=False
    )

    details_wfo = find_column(
        details,
        ["wfo"],
        required=False
    )

    details_episode_id = find_column(
        details,
        ["episode_id"],
        required=False
    )

    details_event_narrative = find_column(
        details,
        ["event_narrative"],
        required=False
    )

    details_episode_narrative = find_column(
        details,
        ["episode_narrative"],
        required=False
    )

    details_damage_property = find_column(
        details,
        ["damage_property"],
        required=False
    )

    details_damage_crops = find_column(
        details,
        ["damage_crops"],
        required=False
    )

    details_injuries_direct = find_column(
        details,
        ["injuries_direct"],
        required=False
    )

    details_injuries_indirect = find_column(
        details,
        ["injuries_indirect"],
        required=False
    )

    details_deaths_direct = find_column(
        details,
        ["deaths_direct"],
        required=False
    )

    details_deaths_indirect = find_column(
        details,
        ["deaths_indirect"],
        required=False
    )

    details_magnitude = find_column(
        details,
        ["magnitude"],
        required=False
    )

    details_magnitude_type = find_column(
        details,
        ["magnitude_type"],
        required=False
    )

    details_tor_f_scale = find_column(
        details,
        ["tor_f_scale"],
        required=False
    )

    details_begin_lat = find_column(
        details,
        ["begin_lat", "begin_latitude"],
        required=False
    )

    details_begin_lon = find_column(
        details,
        ["begin_lon", "begin_longitude"],
        required=False
    )

    # Keep only useful detail fields.
    detail_keep = {
        details_event_id: "event_id"
    }

    optional_detail_columns = {
        details_event_type: "event_type",
        details_begin: "begin_date_time",
        details_end: "end_date_time",
        details_state: "state",
        details_wfo: "wfo",
        details_episode_id: "episode_id",
        details_event_narrative: "event_narrative",
        details_episode_narrative: "episode_narrative",
        details_damage_property: "damage_property",
        details_damage_crops: "damage_crops",
        details_injuries_direct: "injuries_direct",
        details_injuries_indirect: "injuries_indirect",
        details_deaths_direct: "deaths_direct",
        details_deaths_indirect: "deaths_indirect",
        details_magnitude: "magnitude",
        details_magnitude_type: "magnitude_type",
        details_tor_f_scale: "tor_f_scale",
        details_begin_lat: "begin_lat",
        details_begin_lon: "begin_lon",
    }

    for source_name, output_name in optional_detail_columns.items():
        if source_name is not None:
            detail_keep[source_name] = output_name

    details = details[list(detail_keep.keys())].rename(
        columns=detail_keep
    )

    # EVENT_ID should be numeric where possible.
    details["event_id"] = pd.to_numeric(
        details["event_id"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # LOCATION FILE
    # --------------------------------------------------------

    location_event_id = find_column(
        locations,
        ["event_id"],
        required=True
    )

    location_index = find_column(
        locations,
        ["location_index"],
        required=False
    )

    location_name = find_column(
        locations,
        ["location"],
        required=False
    )

    location_lat = find_column(
        locations,
        ["latitude", "lat"],
        required=True
    )

    location_lon = find_column(
        locations,
        ["longitude", "lon"],
        required=True
    )

    location_begin = find_column(
        locations,
        ["begin_date_time", "begin_datetime"],
        required=False
    )

    location_end = find_column(
        locations,
        ["end_date_time", "end_datetime"],
        required=False
    )

    location_range = find_column(
        locations,
        ["range"],
        required=False
    )

    location_azimuth = find_column(
        locations,
        ["azimuth"],
        required=False
    )

    location_keep = {
        location_event_id: "event_id",
        location_lat: "latitude",
        location_lon: "longitude",
    }

    optional_location_columns = {
        location_index: "location_index",
        location_name: "location",
        location_begin: "location_begin_date_time",
        location_end: "location_end_date_time",
        location_range: "location_range",
        location_azimuth: "location_azimuth",
    }

    for source_name, output_name in optional_location_columns.items():
        if source_name is not None:
            location_keep[source_name] = output_name

    locations = locations[list(location_keep.keys())].rename(
        columns=location_keep
    )

    locations["event_id"] = pd.to_numeric(
        locations["event_id"],
        errors="coerce"
    )

    locations["latitude"] = pd.to_numeric(
        locations["latitude"],
        errors="coerce"
    )

    locations["longitude"] = pd.to_numeric(
        locations["longitude"],
        errors="coerce"
    )

    # Remove rows that don't have usable coordinates.
    locations = locations.dropna(
        subset=["event_id", "latitude", "longitude"]
    ).copy()

    # --------------------------------------------------------
    # CALCULATE DISTANCE FROM JENNETTE'S PIER
    # --------------------------------------------------------

    locations["distance_miles"] = haversine_miles(
        PIER_LAT,
        PIER_LON,
        locations["latitude"],
        locations["longitude"]
    )

    nearby_locations = locations[
        locations["distance_miles"] <= RADIUS_MILES
    ].copy()

    print(
        f"\nLocation records within {RADIUS_MILES:.0f} miles: "
        f"{len(nearby_locations):,}"
    )

    if nearby_locations.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # ONE RECORD PER NOAA EVENT
    # --------------------------------------------------------
    #
    # An event can have multiple NOAA location records.
    # Keep the closest reported location for each EVENT_ID.
    # This prevents duplicate events in the main event table.
    # --------------------------------------------------------

    nearby_locations = nearby_locations.sort_values(
        ["event_id", "distance_miles"]
    )

    nearest = nearby_locations.drop_duplicates(
        subset=["event_id"],
        keep="first"
    ).copy()

    # Count how many qualifying location records each event has.
    location_counts = (
        nearby_locations
        .groupby("event_id")
        .size()
        .rename("nearby_location_count")
        .reset_index()
    )

    nearest = nearest.merge(
        location_counts,
        on="event_id",
        how="left"
    )

    # --------------------------------------------------------
    # JOIN DETAILS
    # --------------------------------------------------------

    merged = nearest.merge(
        details,
        on="event_id",
        how="left"
    )

    # --------------------------------------------------------
    # DATE FILTER
    # --------------------------------------------------------

    if "begin_date_time" in merged.columns:
        merged["begin_date_time"] = pd.to_datetime(
            merged["begin_date_time"],
            errors="coerce"
        )

        merged = merged[
            merged["begin_date_time"].isna()
            | (merged["begin_date_time"] >= START_DATE)
        ].copy()

    # --------------------------------------------------------
    # ADD PROJECT FIELDS
    # --------------------------------------------------------

    merged["project_name"] = "Jennette's Pier NOAA Storm Events"

    merged["reference_location"] = PIER_NAME

    merged["reference_latitude"] = PIER_LAT

    merged["reference_longitude"] = PIER_LON

    merged["radius_miles"] = RADIUS_MILES

    merged["distance_miles"] = pd.to_numeric(
        merged["distance_miles"],
        errors="coerce"
    ).round(3)

    # Year from event start.
    if "begin_date_time" in merged.columns:
        merged["year"] = merged["begin_date_time"].dt.year
        merged["month"] = merged["begin_date_time"].dt.month
        merged["month_name"] = merged["begin_date_time"].dt.month_name()

    print(
        f"Unique NOAA events within radius after date filter: "
        f"{len(merged):,}"
    )

    return merged


# ============================================================
# BUILD ALL YEARS
# ============================================================

def build_dataset():
    all_years = []

    for year in YEARS:
        try:
            result = process_year(year)

            if not result.empty:
                all_years.append(result)

        except Exception as exc:
            print("\n" + "!" * 70)
            print(f"ERROR PROCESSING {year}")
            print("!" * 70)
            print(str(exc))
            raise

    if not all_years:
        raise RuntimeError(
            "No NOAA storm events were found within the project radius."
        )

    data = pd.concat(
        all_years,
        ignore_index=True
    )

    # Remove duplicate EVENT_IDs across years, just in case.
    data = data.drop_duplicates(
        subset=["event_id"],
        keep="first"
    ).copy()

    # Sort chronologically.
    if "begin_date_time" in data.columns:
        data = data.sort_values(
            "begin_date_time",
            na_position="last"
        )

    data = data.reset_index(drop=True)

    return data


# ============================================================
# EXCEL OUTPUT
# ============================================================

def write_excel(data):
    print("\nCreating Excel workbook...")

    # Make a clean copy.
    events = data.copy()

    # Human-friendly column order.
    preferred_columns = [
        "event_id",
        "episode_id",
        "event_type",
        "begin_date_time",
        "end_date_time",
        "state",
        "wfo",
        "location",
        "latitude",
        "longitude",
        "distance_miles",
        "nearby_location_count",
        "magnitude",
        "magnitude_type",
        "tor_f_scale",
        "begin_lat",
        "begin_lon",
        "damage_property",
        "damage_crops",
        "injuries_direct",
        "injuries_indirect",
        "deaths_direct",
        "deaths_indirect",
        "event_narrative",
        "episode_narrative",
    ]

    columns = [
        c for c in preferred_columns
        if c in events.columns
    ]

    # Add any remaining fields afterward.
    remaining = [
        c for c in events.columns
        if c not in columns
    ]

    events = events[columns + remaining]

    # Timeline.
    timeline = events.copy()

    # Monthly summary.
    if "begin_date_time" in events.columns:
        monthly = (
            events
            .assign(
                month_period=events["begin_date_time"].dt.to_period("M")
            )
            .groupby("month_period")
            .size()
            .reset_index(name="event_count")
        )

        monthly["month"] = monthly["month_period"].astype(str)

        monthly = monthly[
            ["month", "event_count"]
        ]

    else:
        monthly = pd.DataFrame(
            columns=["month", "event_count"]
        )

    # Event type summary.
    if "event_type" in events.columns:
        event_types = (
            events["event_type"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("event_type")
            .reset_index(name="event_count")
        )

    else:
        event_types = pd.DataFrame(
            columns=["event_type", "event_count"]
        )

    # Map data.
    map_columns = [
        c for c in [
            "event_id",
            "event_type",
            "begin_date_time",
            "location",
            "latitude",
            "longitude",
            "distance_miles",
            "magnitude",
            "magnitude_type",
        ]
        if c in events.columns
    ]

    map_data = events[map_columns].copy()

    # Project setup.
    project_setup = pd.DataFrame({
        "Setting": [
            "Project",
            "Reference Location",
            "Latitude",
            "Longitude",
            "Radius (statute miles)",
            "Start Date",
            "NOAA Source",
            "Years Processed",
        ],
        "Value": [
            "Jennette's Pier NOAA Storm Events",
            PIER_NAME,
            PIER_LAT,
            PIER_LON,
            RADIUS_MILES,
            START_DATE.strftime("%Y-%m-%d"),
            NOAA_BASE,
            ", ".join(str(y) for y in YEARS),
        ]
    })

    # README.
    readme = pd.DataFrame({
        "Item": [
            "Purpose",
            "Reference point",
            "Radius",
            "Start date",
            "Event definition",
            "Location handling",
            "Primary data source",
        ],
        "Description": [
            "NOAA/NCEI Storm Events occurring at reported locations within the project radius.",
            f"{PIER_NAME} ({PIER_LAT}, {PIER_LON})",
            f"{RADIUS_MILES} statute miles",
            START_DATE.strftime("%Y-%m-%d"),
            "One row per NOAA EVENT_ID.",
            "If an event has multiple reported locations inside the radius, the closest reported location is used for the main event record.",
            "NOAA/NCEI Storm Events Database bulk CSV files.",
        ]
    })

    with pd.ExcelWriter(
        EXCEL_FILE,
        engine="openpyxl"
    ) as writer:

        readme.to_excel(
            writer,
            sheet_name="README",
            index=False
        )

        events.to_excel(
            writer,
            sheet_name="All Events",
            index=False
        )

        timeline.to_excel(
            writer,
            sheet_name="Timeline",
            index=False
        )

        monthly.to_excel(
            writer,
            sheet_name="Monthly Summary",
            index=False
        )

        event_types.to_excel(
            writer,
            sheet_name="Event Types",
            index=False
        )

        map_data.to_excel(
            writer,
            sheet_name="Map Data",
            index=False
        )

        project_setup.to_excel(
            writer,
            sheet_name="Project Setup",
            index=False
        )

    print(f"\nExcel created:")
    print(EXCEL_FILE)


# ============================================================
# JSON / JAVASCRIPT OUTPUT
# ============================================================

def write_dashboard_data(data):
    """
    Create JSON and JavaScript files for the interactive dashboard.
    """

    print("\nCreating dashboard data files...")

    records = []

    for _, row in data.iterrows():

        record = {}

        for column in data.columns:
            value = clean_value(row[column])
            record[column] = value

        records.append(record)

    # JSON.
    with open(
        JSON_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False
        )

    # JavaScript.
    #
    # This can be loaded by index.html with:
    #
    # <script src="Jennettes_Pier_Weather_Data.js"></script>
    #
    # and then the dashboard can use WEATHER_DATA.
    js_text = (
        "const WEATHER_DATA = "
        + json.dumps(
            records,
            indent=2,
            ensure_ascii=False
        )
        + ";"
    )

    with open(
        JS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(js_text)

    print(f"JSON created:")
    print(JSON_FILE)

    print(f"\nJavaScript data created:")
    print(JS_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("JENNETTE'S PIER NOAA STORM EVENTS PROJECT")
    print("=" * 70)
    print(f"Reference: {PIER_NAME}")
    print(f"Coordinates: {PIER_LAT}, {PIER_LON}")
    print(f"Radius: {RADIUS_MILES} statute miles")
    print(f"Start date: {START_DATE.date()}")
    print("=" * 70)

    data = build_dataset()

    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print(f"Total NOAA events: {len(data):,}")

    if "event_type" in data.columns:
        print("\nEvent types:")
        print(
            data["event_type"]
            .fillna("Unknown")
            .value_counts()
            .to_string()
        )

    if "distance_miles" in data.columns:
        print("\nClosest event:")
        closest = data.loc[
            data["distance_miles"].idxmin()
        ]

        print(
            f"Event ID: {closest.get('event_id')}\n"
            f"Type: {closest.get('event_type')}\n"
            f"Location: {closest.get('location')}\n"
            f"Distance: {closest.get('distance_miles')} miles"
        )

    write_excel(data)

    write_dashboard_data(data)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nFiles created in:")
    print(OUTPUT_DIR)

    print("\n1. Jennettes_Pier_NOAA_Weather_Events.xlsx")
    print("2. Jennettes_Pier_Weather_Data.json")
    print("3. Jennettes_Pier_Weather_Data.js")

    print("\nYou can now connect the JavaScript data file to the dashboard.")


if __name__ == "__main__":
    main()
