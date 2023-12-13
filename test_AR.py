import pytest

from AR import AR

def test_invalid_quantity():
    with pytest.raises(Exception) as e_info:
        ar = AR('./data/SG.csv', 'IdoNotExist')
        assert False

def test_input_json():
    with pytest.raises(Exception) as e_info:
        ar = AR('./data/SG.json', 'IdoNotExist')
        assert False

def test_file_not_exists():
    with pytest.raises(Exception) as e_info:
        ar = AR('./data/SG_1.csv', 'IdoNotExist')
        assert False