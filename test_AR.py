import pytest

from AR import AR

def test_invalid_quantity():
    with pytest.raises(Exception) as e_info:
        ar = AR('./data/SG.csv', 'IdoNotExist')
        assert False