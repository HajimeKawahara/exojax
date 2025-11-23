from exojax.test.data import TESTDATA_EXOMOLHR_CSV
from exojax.test.data import get_testdata_filename
from exojax.database.exomolhr._downloader import _load_exomolhr_csv

def test_exomolhr_load():
    csv_path = get_testdata_filename(TESTDATA_EXOMOLHR_CSV)
    df = _load_exomolhr_csv(csv_path)





if __name__ == "__main__":
    test_exomolhr_load()
    print("Test passed.")