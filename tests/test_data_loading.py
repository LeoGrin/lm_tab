from src.data_loading import *

def test_load_data():
    for data_name in ["journal_jcr_cls", "movies", "michelin", "spotify", "employee_salary", "drug_directory", "museums", "fifa_footballplayers_22", "jp_anime"]:
        X, y = load_data(data_name, max_rows=None)
        assert len(X) == len(y)
        X, y = load_data(data_name, max_rows=100)
        assert len(X) == len(y) == 100
