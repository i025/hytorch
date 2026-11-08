import numpy as np
import pytest
from hytorch.backends.cuda import runtime as rt
from hytorch.functions_conv import col2im_array

def test_col2im_default_and_cpu_fallback():
    old=rt.current_config();assert old['col2im']=='cupy'
    x=np.ones((1,1,1,1,2,3),np.float32)
    with rt.backend(col2im='native'):
        np.testing.assert_array_equal(col2im_array(x,(1,1,2,3),1,1,0,False),np.ones((1,1,2,3),np.float32))
    assert rt.current_config()==old
    with pytest.raises(ValueError):
        with rt.backend(col2im='unknown'):pass
