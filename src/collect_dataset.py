import time
from pathlib import Path
import requests
from ddgs import DDGS

RAW_DIR = Path("../data/raw")
CATEGORIES: dict[str, list[str]] = {
    "poterie_nabeul": [
        "poterie ceramique nabeul tunisie",
        "pottery nabeul tunisia",
        "poterie nabeul gros plan motif",
        "Nabeul pottery close up pattern",
        "assiette ceramique tunisienne table",
    ],
    "tapis_kairouan": [
        "tapis kilim kairouan tunisie",
        "carpet kairouan tunisia",
        "tapis berbere texture gros plan laine",
        "Kairouan rug wool texture close up",
        "tapis tunisien sol interieur maison",
    ],
    "broderie_tunisienne": [
        "broderie tunisienne traditionnelle",
        "embroidery tunisia traditional",
        "broderie fil or gros plan",
        "Tunisian embroidery gold thread close up",
        "coussin brode tunisien interieur",
    ],
    "bijoux_berberes": [
        "bijoux berberes tunisie argent",
        "berber jewelry tunisia silver",
        "khamsa main fatma argent gros plan",
        "Berber silver jewelry close up",
        "collier berbere femme portee",
    ],
    "maroquinerie_tunisienne": [
        "maroquinerie cuir tunisie",
        "leather craft tunisia",
        "babouche cuir tunisien gros plan",
        "Tunisia leather babouche close up",
        "sac cuir artisanal souk tunisien",
    ],
    "bois_sculpte": [
        "bois sculpte artisanat tunisien",
        "wood carving tunisia",
        "boite bois sculpte gros plan",
        "olive wood box close up texture",
        "meuble marqueterie tunisien interieur",
    ],
    "verre_souffle": [
        "verre souffle tunisien",
        "blown glass tunisia",
        "verre souffle gros plan couleur",
        "Tunisia blown glass close up color",
        "vase verre artisanal tunisien table",
    ],
    "fer_forge": [
        "fer forge artisanat tunisien",
        "wrought iron tunisia",
        "lanterne fer forge gros plan",
        "wrought iron lantern close up detail",
        "porte fer forge medina ancienne",
    ],
    "cuivre": [
        "dinanderie tunisienne cuivre",
        "Tunisia copper teapot traditional",
        "plateau cuivre tunisien cisele",
        "Tunisia brass tray engraved",
        "theiere cuivre tunisienne",
    ],
    "djebba": [
        "djebba tunisienne traditionnelle homme",
        "Tunisia traditional men jebba garment",
        "jebba brodee tunisienne mariage",
        "Tunisia embroidered ceremonial robe men",
        "tunisien habit traditionnel blanc",
    ],
}

MAX_IMAGES_PER_QUERY = 50
REQUEST_TIMEOUT_SECONDS = 10
MIN_VALID_FILE_SIZE_BYTES = 5_000
PAUSE_BETWEEN_QUERIES_SECONDS = 6
PAUSE_ON_ERROR_SECONDS = 30
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def download_image(url: str, destination: Path) -> bool:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})
        if response.status_code == 200 and len(response.content) > MIN_VALID_FILE_SIZE_BYTES:
            destination.write_bytes(response.content)
            return True
    except requests.RequestException:
        pass
    return False


def collect_category(category: str, queries: list[str], output_dir: Path) -> int:
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    image_count = len(list(category_dir.glob("*.jpg")))
    print(f"\n{category} (existing: {image_count})")

    for query in queries:
        print(f"  query: '{query}'")
        time.sleep(PAUSE_BETWEEN_QUERIES_SECONDS)

        try:
            results = list(DDGS().images(query, max_results=MAX_IMAGES_PER_QUERY))
            print(f"    found {len(results)} candidate URLs")
        except Exception as error:
            print(f"    search failed: {error} — pausing {PAUSE_ON_ERROR_SECONDS}s")
            time.sleep(PAUSE_ON_ERROR_SECONDS)
            continue

        for i, result in enumerate(results):
            url = result.get("image", "")
            if not url:
                continue
            destination = category_dir / f"{category}_{image_count:04d}.jpg"
            if download_image(url, destination):
                image_count += 1
            if i % 10 == 0 and i > 0:
                time.sleep(1)

    print(f"  total for {category}: {image_count} images")
    return image_count


def collect_dataset(categories: dict[str, list[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {category: collect_category(category, queries, output_dir) for category, queries in categories.items()}

    print("\nCollection summary")
    for category, total in totals.items():
        print(f"{category:<28} {total:>5} images")
    print(f"{'total':<28} {sum(totals.values()):>5} images")


if __name__ == "__main__":
    collect_dataset(CATEGORIES, RAW_DIR)