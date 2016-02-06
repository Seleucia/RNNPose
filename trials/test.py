import numpy as np
import helper.utils as u
import theano

def sample_weights(sizeX, sizeY):
    values = np.ndarray([sizeX, sizeY], dtype=float)
    for dx in xrange(sizeX):
        vals = np.random.uniform(low=-1., high=1.,  size=(sizeY,))
        #vals_norm = np.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value
    values = values / svs[0]
    return values


def shared_floatx(value, name=None, borrow=False, dtype=None, **kwargs):
    r"""Transform a value into a shared variable of type floatX.
    Parameters
    ----------
    value : :class:`~numpy.ndarray`
        The value to associate with the Theano shared.
    name : :obj:`str`, optional
        The name for the shared variable. Defaults to `None`.
    borrow : :obj:`bool`, optional
        If set to True, the given `value` will not be copied if possible.
        This can save memory and speed. Defaults to False.
    dtype : :obj:`str`, optional
        The `dtype` of the shared variable. Default value is
        :attr:`config.floatX`.
    \*\*kwargs
        Keyword arguments to pass to the :func:`~theano.shared` function.
    Returns
    -------
    :class:`tensor.TensorSharedVariable`
        A Theano shared variable with the requested value and `dtype`.
    """
    return np.asarray(value, dtype=dtype)

def shared_floatx_nans(shape, **kwargs):
    r"""Creates a shared variable array filled with nans.
    Parameters
    ----------
    shape : tuple
         A tuple of integers representing the shape of the array.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx` function.
    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A Theano shared variable filled with nans.
    """
    return shared_floatx(np.nan * np.zeros(shape), **kwargs)

n_lstm=12

W_cell_to_in = shared_floatx_nans((10,),name='W_cell_to_in')

val=sample_weights(5, 5)
b=u.init_bias(n_lstm, sample='one')
W_cf = u.init_weight((n_lstm, n_lstm),'W_cf', 'svd')
print val