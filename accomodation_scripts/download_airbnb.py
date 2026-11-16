'''from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
import gzip
import shutil
import re

# SETTINGS

URL = "https://insideairbnb.com/get-the-data/"

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "data" / "accommodation" / "raw"

# Detailed listings dataset only
PATTERN = re.compile(
    r"/([^/]+)/(\d{4}-\d{2}-\d{2})/data/listings\.csv\.gz$"
)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

# STEP 1: FIND ALL CITIES AUTOMATICALLY

print("Opening Inside Airbnb...")

response = session.get(URL, timeout=90)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

latest_snapshots = {}

for link in soup.find_all("a", href=True):

    full_url = urljoin(URL, link["href"])
    path = urlparse(full_url).path

    match = PATTERN.search(path)

    if not match:
        continue

    city, date = match.groups()

    # Keep only the latest snapshot for each city
    if (
        city not in latest_snapshots
        or date > latest_snapshots[city][0]
    ):
        latest_snapshots[city] = (date, full_url)

print(f"\nFound {len(latest_snapshots)} cities:")
for city in sorted(latest_snapshots):
    print(f"  {city}: {latest_snapshots[city][0]}")

# STEP 2: DOWNLOAD ALL CITIES

successful = 0
skipped = 0
failed = []

for city in sorted(latest_snapshots):

    date, file_url = latest_snapshots[city]

    folder = BASE_DIR / city
    folder.mkdir(parents=True, exist_ok=True)

    output = folder / f"listings_{date}.csv"
    compressed = folder / f"listings_{date}.csv.gz"
    temporary = folder / f"listings_{date}.csv.part"

    print(f"\nChecking {city} ({date})...")

    # Do not replace existing CSV files
    if output.exists():
        print("Already downloaded. Skipping.")
        skipped += 1
        continue

    try:
        print("Downloading...")

        with session.get(
            file_url,
            stream=True,
            timeout=(30, 300)
        ) as response:

            response.raise_for_status()

            with open(compressed, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        print("Extracting CSV...")

        with gzip.open(compressed, "rb") as source:
            with open(temporary, "wb") as target:
                shutil.copyfileobj(source, target)

        # Only create final CSV after extraction succeeds
        temporary.replace(output)

        # Remove compressed file
        compressed.unlink()

        print(f"Saved: {output.name}")
        successful += 1

    except Exception as error:
        print(f"FAILED: {city}: {error}")
        failed.append(city)

        compressed.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)

# STEP 3: SUMMARY

print("\n===================================")
print("DOWNLOAD SUMMARY")
print("===================================")

print(f"Cities found: {len(latest_snapshots)}")
print(f"New downloads: {successful}")
print(f"Already downloaded: {skipped}")
print(f"Failed downloads: {len(failed)}")

if failed:
    print("\nFailed cities:")
    for city in failed:
        print(f"  - {city}")

print("\nFinished!")
'''