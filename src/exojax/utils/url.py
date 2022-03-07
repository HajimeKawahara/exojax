"""url.

* This module contains various url for downloading. Because url can be changed by the orner of the site, you might need to change it. Or, if you notice the change, it would be great if you send a pull-request.
"""


def url_HITRAN12():
    """return URL for HITRAN 12 parfile.

    Returns:
       URL for HITRAN 12 parfile
    """
    url = u'https://www.cfa.harvard.edu/HITRAN/HITRAN2012/HITRAN2012/By-Molecule/Uncompressed-files/'
    return url


def url_HITRANCIA():
    """return URL for HITRAN CIA ciafile.

    Returns:
       URL for HITRAN CIA file
    """
    url = u'https://hitran.org/data/CIA/'
    return url


def url_HITEMP():
    """return URL for HITEMP bz2 parfile.

    Returns:
       URL for HITEMP bz2 file
    """
    url = u'https://hitran.org/hitemp/data/bzip2format/'
    return url


def url_HITEMP10():
    """return URL for HITEMP2010.

    Returns:
       URL for HITEMP2010 db
    """
    url = u'https://hitran.org/hitemp/data/HITEMP-2010/'
    return url


def url_ExoMol():
    """return URL for ExoMol.

    Returns:
       URL for ExoMol db
    """
    url = u'http://www.exomol.com/db/'
    return url


def url_Exomol_iso(molecule, isotope_full_name):
    """return URL for ExoMol for isotope.

    Returns:
       URL for ExoMol for isotope
    """
    url = u'https://exomol.com/data/molecules/' + \
        str(molecule)+'/'+str(isotope_full_name)
    return url


def url_developer_data():
    """return URL for data in exojax.

    Returns:
       URL for ExoJAX
    """
    url = u'http://secondearths.sakura.ne.jp/exojax/data/'
    return url
