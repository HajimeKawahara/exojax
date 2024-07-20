"""url.

* This module contains various url for downloading. Because url can be changed by the orner of the site, you might need to change it. Or, if you notice the change, it would be great if you send a pull-request.
"""


def url_virga():
    """returns URL for VIRGA refractive index data from ZENODO

    Returns:
        URL for VIRGA refractive index data
    """
    url = "https://zenodo.org/records/5179187/files/virga.zip"
    return url


def url_HITRAN12():
    """returns URL for HITRAN 12 parfile.

    Returns:
        URL for HITRAN 12 parfile
    """
    url = "https://www.cfa.harvard.edu/HITRAN/HITRAN2012/HITRAN2012/By-Molecule/Uncompressed-files/"
    return url


def url_HITRANCIA():
    """returns URL for HITRAN CIA ciafile.

    Returns:
        URL for HITRAN CIA file
    """
    url = "https://hitran.org/data/CIA/"
    return url


def url_HITEMP():
    """returns URL for HITEMP bz2 parfile.

    Returns:
        URL for HITEMP bz2 file
    """
    url = "https://hitran.org/hitemp/data/bzip2format/"
    return url


def url_HITEMP10():
    """returns URL for HITEMP2010.

    Returns:
        URL for HITEMP2010 db
    """
    url = "https://hitran.org/hitemp/data/HITEMP-2010/"
    return url


def url_ExoMol():
    """returns URL for ExoMol.

    Returns:
        URL for ExoMol db
    """
    url = "http://www.exomol.com/db/"
    return url


def url_Exomol_iso(molecule, isotope_full_name):
    """returns URL for ExoMol for isotope.

    Returns:
        URL for ExoMol for isotope
    """
    url = (
        "https://exomol.com/data/molecules/"
        + str(molecule)
        + "/"
        + str(isotope_full_name)
    )
    return url


def url_developer_data():
    """returns URL for data in exojax.

    Returns:
        URL for ExoJAX
    """
    url = "http://secondearths.sakura.ne.jp/exojax/data/"
    return url
