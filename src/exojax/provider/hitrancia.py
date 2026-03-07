from pathlib import Path
import urllib.request
from exojax.provider.url import url_HITRANCIA


def fetch_hitran_cia(path):
    """Downloading hitrancia file.

    Note:
        The download URL is written in exojax.provider.url.
    """

    out_dir = Path(path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading HITRAN CIA data...")
        url = url_HITRANCIA() + path.name
        data = urllib.request.urlopen(url).read()
        with open(str(path), mode="wb") as f:
            f.write(data)
        # urllib.request.urlretrieve(url, str(self.path))
    except:
        print(
            "HITRAN CIA download failed. Please download it manually from HITRAN website."
        )

