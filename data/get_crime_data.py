import requests
import zipfile
import io
import pandas as pd
import os
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    """Minimal .env loader for KEY=VALUE lines."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env_file(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("DATA_GOV_API_KEY", "")
BASE = "https://api.usa.gov/crime/fbi/cde"

if not API_KEY:
    raise ValueError("Missing DATA_GOV_API_KEY in .env")

# ── City-level: Offenses Known to Law Enforcement ──────────────────────────
# This is your primary merge key for Airbnb city matching
# Returns all agencies/cities for all states, 2024 (most recent available)

def get_offenses_by_city(year=2024):
    """Pull offenses known to law enforcement at agency/city level."""
    url = f"{BASE}/summarized/agency/offenses/{year}/all"
    params = {"per_page": 10000, "page": 1, "api_key": API_KEY}
    all_data = []
    while True:
        r = requests.get(url, params=params)
        data = r.json()
        results = data.get("results", [])
        all_data.extend(results)
        if len(results) < 10000:
            break
        params["page"] += 1
    return pd.DataFrame(all_data)

# ── State-level ─────────────────────────────────────────────────────────────
def get_offenses_by_state(year=2024):
    """Pull summarized offense totals at the state level."""
    states = [
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
        "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
        "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
        "TX","UT","VT","VA","WA","WV","WI","WY","DC"
    ]
    all_data = []
    for state in states:
        url = f"{BASE}/summarized/state/{state}/offenses/{year}/{year}"
        r = requests.get(url, params={"api_key": API_KEY})
        data = r.json()
        if "results" in data:
            for row in data["results"]:
                row["state_abbr"] = state
            all_data.extend(data["results"])
    return pd.DataFrame(all_data)

# ── Download bulk ZIP directly (alternative) ───────────────────────────────
def download_bulk_zip():
    """
    The CDE bulk file is also available at this endpoint.
    Requires a valid API key.
    """
    url = f"{BASE}/downloads/offenses-known-to-law-enforcement/2024"
    r = requests.get(url, params={"api_key": API_KEY}, stream=True)
    with open("offenses_known_2024.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Downloaded offenses_known_2024.zip")

# ── Run it ──────────────────────────────────────────────────────────────────
city_df = get_offenses_by_city(2024)
city_df.to_csv("fbi_offenses_by_city_2024.csv", index=False)
print(f"City-level: {len(city_df)} rows saved")

state_df = get_offenses_by_state(2024)
state_df.to_csv("fbi_offenses_by_state_2024.csv", index=False)
print(f"State-level: {len(state_df)} rows saved")