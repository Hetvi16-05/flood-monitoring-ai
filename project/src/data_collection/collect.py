import os
import requests
import time
import uuid
import hashlib
from tqdm import tqdm


# =========================
# API KEYS
# =========================

PEXELS_KEY = "W1SAen94jRy0ZDSjjjoF6UbLhp56bfy9d8sV7NOd2o8G5fQsIFybyeRu"
UNSPLASH_KEY = "fGE0EjOaLy_XZR35w4ETHHou85ETlyITBJ2fhGHkH8Y"
PEXELS_URL = "https://api.pexels.com/v1/search"
UNSPLASH_URL = "https://api.unsplash.com/search/photos"


# =========================
# SETTINGS
# =========================

BASE_DIR = "project/raw_dataset"

TARGET = 7500

PER_PAGE_PEXELS = 80
PER_PAGE_UNSPLASH = 30

DELAY = 1


# =========================
# QUERIES
# =========================

QUERIES = {

      "building": [
        "building",
        "house",
        "apartment",
        "city building",
        "urban",
        "architecture"
    ]
}

# =========================
# HASH
# =========================

def get_hash(data):
    return hashlib.md5(data).hexdigest()


def load_hashes(folder):

    hashes = set()

    if not os.path.exists(folder):
        return hashes

    for f in os.listdir(folder):

        path = os.path.join(folder, f)

        try:
            with open(path, "rb") as fp:
                hashes.add(get_hash(fp.read()))
        except:
            pass

    return hashes


# =========================
# COUNT
# =========================

def count_images(folder):

    if not os.path.exists(folder):
        return 0

    return len([
        f for f in os.listdir(folder)
        if f.endswith(".jpg")
    ])


# =========================
# FETCH PEXELS
# =========================

def fetch_pexels(query, need):

    headers = {
        "Authorization": PEXELS_KEY
    }

    urls = []
    page = 1

    while len(urls) < need:

        params = {
            "query": query,
            "page": page,
            "per_page": PER_PAGE_PEXELS
        }

        r = requests.get(PEXELS_URL, headers=headers, params=params)

        if r.status_code != 200:
            break

        data = r.json()

        results = data.get("photos", [])

        if not results:
            break

        for item in results:

            urls.append(item["src"]["large"])

            if len(urls) >= need:
                break

        page += 1
        time.sleep(DELAY)

    return urls


# =========================
# FETCH UNSPLASH
# =========================

def fetch_unsplash(query, need):

    headers = {
        "Authorization": f"Client-ID {UNSPLASH_KEY}"
    }

    urls = []
    page = 1

    while len(urls) < need:

        params = {
            "query": query,
            "page": page,
            "per_page": PER_PAGE_UNSPLASH
        }

        r = requests.get(UNSPLASH_URL, headers=headers, params=params)

        if r.status_code != 200:
            break

        data = r.json()

        results = data["results"]

        if not results:
            break

        for item in results:

            urls.append(item["urls"]["regular"])

            if len(urls) >= need:
                break

        page += 1
        time.sleep(DELAY)

    return urls


# =========================
# DOWNLOAD
# =========================

def download(url, folder, hashes):

    try:

        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return False

        h = get_hash(r.content)

        if h in hashes:
            return False

        name = uuid.uuid4().hex + ".jpg"

        path = os.path.join(folder, name)

        with open(path, "wb") as f:
            f.write(r.content)

        hashes.add(h)

        return True

    except:
        return False


# =========================
# MAIN
# =========================

def main():

    print("🚀 Starting Data Collection")

    for cls, queries in QUERIES.items():

        folder = os.path.join(BASE_DIR, cls)

        os.makedirs(folder, exist_ok=True)

        hashes = load_hashes(folder)

        current = count_images(folder)

        print("\nClass:", cls, "| Current:", current)

        need = TARGET - current

        if need <= 0:
            continue

        # divide per query
        per_query = need // len(queries) + 1

        for q in queries:

            if need <= 0:
                break

            print("  Fetching:", q)

            urls = []

            urls += fetch_pexels(q, per_query)
            urls += fetch_unsplash(q, per_query)

            if len(urls) == 0:
                print("  ⚠️ No images found")
                continue

            for url in tqdm(urls, desc=f"Downloading {q}"):

                if download(url, folder, hashes):
                    need -= 1

                if need <= 0:
                    break


if __name__ == "__main__":
    main()