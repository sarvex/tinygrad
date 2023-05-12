import numpy as np
from .tensor import Function

# ************* unary ops *************

class ReLU(Function):
  def forward(self, input):
    self.save_for_backward(input)
    return np.maximum(input, 0)

  def backward(self, grad_output):
    input, = self.saved_tensors
    return grad_output * (input >= 0)

class Log(Function):
  def forward(self, input):
    self.save_for_backward(input)
    return np.log(input)

  def backward(self, grad_output):
    input, = self.saved_tensors
    return grad_output / input

class Exp(Function):
  def forward(self, input):
    ret = np.exp(input)
    self.save_for_backward(ret)
    return ret

  def backward(self, grad_output):
    ret, = self.saved_tensors
    return grad_output * ret

# ************* reduce ops *************

class Sum(Function):
  def forward(self, input, axis=None):
    self.save_for_backward(input, axis)
    return np.array([input.sum()]) if axis is None else input.sum(axis=axis)

  def backward(self, grad_output):
    input, axis = self.saved_tensors
    if isinstance(axis, int): axis = [axis]
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    return grad_output.reshape(shape) + np.zeros_like(input)

class Max(Function):
  def forward(self, inp, axis=None):
    if isinstance(axis, int): axis = [axis]
    ret = np.amax(inp, axis=None if axis is None else tuple(axis), keepdims=True)
    self.save_for_backward(inp, axis, ret)
    if axis is not None:
      ret = ret.reshape([inp.shape[i] for i in range(len(inp.shape)) if i not in axis])
    return ret

  def backward(self, grad_output):
    input, axis, ret = self.saved_tensors
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    ret2 = (input==ret.reshape(shape))
    div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True)
    return ret2*grad_output.reshape(shape)/div

# ************* binary ops *************

def unbroadcast(out, in_sh):
  # adjoint operation to broadcast is sum. Need to sum all axis with 1 = in_sh[i] < out.shape[i]
  sum_axis = (tuple(i for i in range(len(in_sh))
                    if in_sh[i] == 1 and out.shape[i] > 1) if in_sh !=
              (1, ) else None)
  return out.sum(axis=sum_axis).reshape(in_sh)

class Add(Function):
  def forward(self, x, y):
    self.save_for_backward(x.shape, y.shape)
    return x+y

  def backward(self, grad_output):
    shape_x, shape_y = self.saved_tensors
    return unbroadcast(grad_output, shape_x), unbroadcast(grad_output, shape_y)

class Sub(Function):
  def forward(self, x, y):
    self.save_for_backward(x.shape, y.shape)
    return x-y

  def backward(self, grad_output):
    shape_x, shape_y = self.saved_tensors
    return unbroadcast(grad_output, shape_x), unbroadcast(-grad_output, shape_y)

class Mul(Function):
  def forward(self, x, y):
    self.save_for_backward(x, y)
    return x*y

  def backward(self, grad_output):
    x,y = self.saved_tensors
    return unbroadcast(y*grad_output, x.shape), unbroadcast(x*grad_output, y.shape)

class Pow(Function):
  def forward(self, x, y):
    self.save_for_backward(x, y)
    return x ** y

  def backward(self, grad_output):
    x,y = self.saved_tensors
    return unbroadcast(y * (x**(y-1.0)) * grad_output, x.shape), \
             unbroadcast((x**y) * np.log(x) * grad_output, y.shape)

# ************* movement ops *************

class Reshape(Function):
  def forward(self, x, shape):
    self.save_for_backward(x.shape)
    return x.reshape(shape)

  def backward(self, grad_output):
    in_shape, = self.saved_tensors
    return grad_output.reshape(in_shape)

class Transpose(Function):
  def forward(self, x, order):
    self.save_for_backward(order)
    return np.transpose(x, order)

  def backward(self, x):
    return np.transpose(x, np.argsort(self.order))

def inner_slice(x, arg):
  padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i])) for i,p in enumerate(arg)]
  x = np.pad(x, padding)
  slicee = [(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg)]
  return x[tuple(slice(x[0], x[1], None) for x in slicee)]

class Slice(Function):
  def forward(self, x, arg=None):
    self.save_for_backward(x.shape)
    return inner_slice(x, arg)

  def backward(self, grad_output):
    shape, = self.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(ctx.arg)]
    return inner_slice(grad_output, narg)

# ************* processing ops *************

class Matmul(Function):
  def forward(self, input, weight):
    self.save_for_backward(input, weight)
    return input @ weight

  def backward(self, grad_output):
    input, weight = self.saved_tensors
    grad_input = grad_output @ np.swapaxes(weight, -2, -1)
    grad_weight = np.swapaxes(input, -2, -1) @ grad_output
    return grad_input, grad_weight

class Conv2D(Function):
  def forward(self, x, w, stride=1, groups=1):
    if isinstance(self.stride, int):
      self.stride = self.stride, self.stride
    cout,cin,H,W = w.shape
    ys,xs = self.stride
    bs,cin_ = x.shape[0], x.shape[1]
    oy,ox = (x.shape[2]-(H-ys))//ys, (x.shape[3]-(W-xs))//xs
    assert cin * self.groups == cin_
    assert cout % self.groups == 0
    rcout = cout // self.groups

    gx = x.reshape(bs, self.groups, cin, x.shape[2], x.shape[3])
    tx = np.lib.stride_tricks.as_strided(
        gx,
        shape=(bs, self.groups, cin, oy, ox, H, W),
        strides=(
            *gx.strides[:3],
            gx.strides[3] * ys,
            gx.strides[4] * xs,
            *gx.strides[3:5],
        ),
        writeable=False,
    )
    tw = w.reshape(self.groups, rcout, cin, H, W)
    self.save_for_backward(tx, tw, x.shape)

    ret = np.zeros((bs, self.groups, oy, ox, rcout), dtype=x.dtype)
    for g in range(self.groups):
      #ijYXyx,kjyx -> iYXk ->ikYX
      ret[:,g] += np.tensordot(tx[:,g], tw[g], ((1,4,5),(1,2,3)))
    return np.moveaxis(ret,4,2).reshape(bs, cout, oy, ox)

  def backward(self, grad_output):
    bs,_,oy,ox = grad_output.shape
    tx, tw, x_shape = self.saved_tensors
    _,rcout,cin,H,W = tw.shape
    ys,xs = self.stride
    OY,OX = x_shape[2:4]

    ggg = grad_output.reshape(bs, self.groups, rcout, oy, ox)

    gdw = np.zeros((self.groups, rcout, cin, H, W), dtype=tx.dtype)
    for g in range(self.groups):
      #'ikYX,ijYXyx -> kjyx'
      gdw[g] += np.tensordot(ggg[:,g], tx[:,g], ((0,2,3),(0,2,3)))

    # needs to be optimized
    gdx = np.zeros((bs, self.groups, cin, OY, OX), dtype=tx.dtype)
    for k in range(oy*ox):
      Y, X = k//ox, k%ox
      iY,iX = Y*ys, X*xs
      #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
      for g in range(self.groups):
        tg = np.dot(ggg[:,g,:,Y,X].reshape(bs, -1), tw[g].reshape(rcout, -1))
        gdx[:, g, :, iY:iY+H, iX:iX+W] += tg.reshape((bs, cin, H, W))

    return gdx.reshape((bs, self.groups * cin, OY, OX)), gdw.reshape(
        (self.groups * rcout, cin, H, W))
