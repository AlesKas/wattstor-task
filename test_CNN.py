import pytest

from CNN import CNN

# This is kidna trivial test, 
def test_invalid_quantity():
    with pytest.raises(Exception) as e_info:
        cnn = CNN('./data/SG.csv', 'IdoNotExist')
        assert False


def test_input_json():
    with pytest.raises(Exception) as e_info:
        ar = CNN('./data/SG.json', 'IdoNotExist')
        assert False

def test_file_not_exists():
    with pytest.raises(Exception) as e_info:
        ar = CNN('./data/SG_1.csv', 'IdoNotExist')
        assert False