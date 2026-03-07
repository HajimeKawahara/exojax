# self.path, self.condensate
from pathlib import Path


def download_and_unzip(path, condensate):
    """Downloading virga refractive index data

    Args:   
        path: Path to save the virga data
        condensate: condensate name, such as NH3, H2O, MgSiO3 etc

    Returns:
        virga_condensates: list of available condensates in virga
        refrind_path: Path to the refractive index file of the requested condensate

    Note:
        The download URL is written in exojax.provider.url.
    """
    import os
    import shutil
    import urllib.request

    from exojax.utils.files import (
        find_files_by_extension,
        get_file_names_without_extension,
    )
    from exojax.provider.url import url_virga

    try:
        os.makedirs(str(path), exist_ok=True)
        filepath = path / "virga.zip"
        if (filepath).exists():
            print(
                str(filepath),
                " exists. Remove it if you wanna re-download and unzip.",
            )
        else:
            print("Downloading ", url_virga())
            # urllib.request.urlretrieve(url_virga(), str(filepath))
            data = urllib.request.urlopen(url_virga()).read()
            with open(str(filepath), mode="wb") as f:
                f.write(data)
            shutil.unpack_archive(str(filepath), str(path))
        virga_condensates = get_file_names_without_extension(
            find_files_by_extension(str(path), ".refrind")
        )
        if condensate in virga_condensates:
            refrind_path = path / Path(condensate + ".refrind")
            print("Refractive index file found: ", refrind_path)
        else:
            print(
                "No refrind file found. Refractive indices of ",
                virga_condensates,
                "are available.",
            )
    except:
        print("VIRGA refractive index download failed")

    return virga_condensates, refrind_path
