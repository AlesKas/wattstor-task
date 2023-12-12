import pytest

from CNN import CNN

# This is kidna trivial test, 
def test_invalid_quantity():
    with pytest.raises(Exception) as e_info:
        cnn = CNN('./data/SG.csv', 'IdoNotExist')
        assert False