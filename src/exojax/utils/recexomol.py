"""Get Recommendation from ExoMol."""
from urllib.request import HTTPError, urlopen
from bs4 import BeautifulSoup


def get_exomol_database_list(molecule, isotope_full_name):
    """Parse ExoMol website and return list of available databases, and
    recommended database.

    Args:
       molecule: str
       isotope_full_name: str, isotope full name (ex. ``12C-1H4`` for CH4,1). Get it from

    Returns:
       database list
       database recomendation

    Example:
        databases, recommended = get_exomol_database_list("CH4", "12C-1H4")
        >>> ['xsec-YT10to10', 'YT10to10', 'YT34to10'], 'YT34to10'

    Note: 
       This function is borrowed from radis (https://github.com/radis/radis by @erwanp). See https://github.com/radis/radis/issues/319 in detail.
    """
    from exojax.utils.url import url_Exomol_iso
    url = url_Exomol_iso(molecule, isotope_full_name)
    try:
        response = urlopen(url).read()
    except HTTPError as err:
        raise ValueError(f'HTTPError opening url={url}') from err

    soup = BeautifulSoup(
        response, features='lxml'
    )  # make soup that is parse-able by bs

    # Recommended database
    rows = soup.find_all(
        'a', {'class': 'list-group-item link-list-group-item recommended'}
    )
    databases_recommended = [r.get_attribute_list('title')[0] for r in rows]

    # All others
    rows = soup.find_all(
        'a', {'class': 'list-group-item link-list-group-item'})
    databases = [r.get_attribute_list('title')[0] for r in rows]

    assert len(databases_recommended) <= 1

    databases = databases + databases_recommended

    return databases, databases_recommended[0]


if __name__ == '__main__':
    db, db0 = get_exomol_database_list('CO', '12C-16O')
    assert db0 == 'Li2015'
