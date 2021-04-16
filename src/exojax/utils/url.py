"""url
   * This module contains various url for downloading. Because url can be changed by the orner of the site, you might need to change it. Or, if you notice the change, it would be great if you send a pull-request.

"""

def url_HITRAN12():
    """return URL for HITRAN 12 parfile
    """
    url=u"https://www.cfa.harvard.edu/HITRAN/HITRAN2012/HITRAN2012/By-Molecule/Uncompressed-files/"
    return url

def url_HITRANCIA():
    """return URL for HITRAN CIA ciafile
    """
    url=u"https://hitran.org/data/CIA/"
    return url


def url_HITEMP():
    """return URL for HITEMP bz2 parfile
    """
    url=u"https://hitran.org/hitemp/data/bzip2format/"
    return url

def url_ExoMol():
    """return URL for ExoMol
    """
    url=u"http://www.exomol.com/db/"
    return url
