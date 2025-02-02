import torch as t
import torch.nn as nn
import lab as B
import lab.torch
from .conv_mm import conv_mm
from .abstract_bnn import AbstractLinear, AbstractConv2d
from .lop import mvnormal_log_prob_unnorm
from .priors import NealPrior, InsanePrior


def rsample_logpq_weights(self, XLX, XLY, prior, neuron_prec=True):
    device = XLX.device
    in_features = XLX.shape[-1] # Din
    out_features = XLY.shape[-3] # Dout

    # X : S x Din x N
    # Y : Dout x N x 1
    # y are the variational parameter, one value for the entire batch samples
    
    assert 4 == len(XLX.shape) and 4 == len(XLY.shape)
    assert XLX.shape[0] == XLY.shape[0] # sample dimensions are the same
    assert XLX.shape[-3] in (out_features, 1) # number of output features, or 1 meaning same precision
    assert XLY.shape[-1] == 1 

    S = XLX.shape[0] # mini batch size
    prior_prec = prior(S)

    prior_prec_full = prior_prec.full()
    if len(prior_prec_full.shape) > 2:
        prior_prec_full = prior_prec_full.unsqueeze(1)

    prec = XLX + prior_prec_full # precision matrix
    L = t.linalg.cholesky(prec) # lower triangular decomp of the Hermitian PD precision matrix 
    # L = B.chol(prec)

    logdet_prec = 2*L.diagonal(dim1=-1, dim2=-2).log().sum(-1)
    logdet_prec = logdet_prec.expand(S, out_features).sum(-1)

    if self._sample is not None:  # if sample drawn, need to figure out what the Z was
        assert S==self._sample.shape[0]
        dW = self._sample.unsqueeze(-1) - t.cholesky_solve(XLY, L)
        Z = L.transpose(-1, -2) @ dW
    else: 
        self.key, Z = B.randn(self.key, B.default_dtype, S, out_features, in_features, 1)
        
        # transform Z to weight space; L is chol of precision
        dW, _ = t.triangular_solve(Z, L, upper=False, transpose=True) # (b, A)
        # dW = B.triangular_solve(L.transpose(-1, -2), Z, lower_a=True) # (A.T, b)
 
        # sample = mu + dW = (LL^T)^-1 @ XLY + (L^-1)@(eps) = var @ XLY + chol(var)@eps
        self._sample = (t.cholesky_solve(XLY, L) + dW).squeeze(-1) # reparameterization of the ELBO to be a function of the noise/sample

    # have a weight sample, compute KL divergence
    logP = mvnormal_log_prob_unnorm(prior_prec, self._sample.transpose(-1, -2))
    logQ = -0.5*(Z**2).sum((-1, -2, -3)) + 0.5*logdet_prec

    logPQw = logP-logQ
    self.logpq = logPQw
    # print(self.log_prec_scaled.exp())
    # print(self.prec_L)
    return self._sample


def rsample_logpq_weights_fc(self, Xi, neuron_prec):
    log_prec = self.log_prec_lr*self.log_prec_scaled
    XiT = Xi.transpose(-1, -2)
    if self.prec_L is None:
        XiLT  = log_prec.exp() * XiT#.transpose(-1, -2)
    else:
        XiLT = XiT @ ((self.prec_L*log_prec.exp())@self.prec_L.transpose(-1, -2))
        # print("Lam:")
        # print(((self.prec_L*log_prec.exp())@self.prec_L.transpose(-1, -2)))
    XiLXi = XiLT @ Xi
    XiLY  = XiLT @ self.u
    return rsample_logpq_weights(self, XLX=XiLXi, XLY=XiLY, prior=self.prior, neuron_prec=neuron_prec)


class GILinearWeights(nn.Module):
    def __init__(self, in_features, out_features, key=None, prior=NealPrior, bias=True, inducing_targets=None, log_prec_init=-4., log_prec_lr=1., neuron_prec=False, inducing_batch=None, full_prec=False):
        super().__init__()
        
        # Set inducing points
        assert inducing_batch is not None
        self.inducing_batch = inducing_batch

        # Init layer shape
        self.in_features = in_features
        self.out_features = out_features
        self.key = key
        self.bias = bias

        # Init prior
        in_shape   = t.Size([in_features+bias])
        self.prior = prior(in_shape)

        # Set initial variational precision parameter: Lambda_l
        self.log_prec_init = log_prec_init
        self.log_prec_lr = log_prec_lr # multiplier for the learning rate of precision pawrameter
        lp_init = self.log_prec_init / self.log_prec_lr # initial log precision
        self.neuron_prec = neuron_prec # different precision param for every neuron => expensive (little gain)
        precs = out_features if neuron_prec else 1
        self.log_prec_scaled = nn.Parameter(lp_init*t.ones(precs, 1, inducing_batch)) # create variational param in nn graph
        
        if full_prec:
            self.prec_L = nn.Parameter(B.eye(B.default_dtype, inducing_batch).expand(precs, -1, -1))
        else:
            self.prec_L = None
        
        if inducing_targets is None:
            # self.u = nn.Parameter(B.randn(self.key, self.out_features, inducing_batch, 1))
            self.u = nn.Parameter(B.randn(self.out_features, inducing_batch, 1))
        else:
            self.u = nn.Parameter(inducing_targets.clone().to(B.default_dtype).transpose(-1, -2).unsqueeze(-1))

        
        
        self._sample = None

    def forward(self, X):
        Xi = X[:, :self.inducing_batch, :]
        return rsample_logpq_weights_fc(self, Xi.unsqueeze(1), neuron_prec=True)


class LILinearWeights(nn.Module):
    def __init__(self, in_features, out_features, prior=NealPrior, bias=True, log_prec_init=-4., log_prec_lr=1., neuron_prec=False, full_prec=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        in_shape   = t.Size([in_features+bias])
        self.prior = prior(in_shape)
        self.neuron_prec = neuron_prec

        inducing_batch = in_features + bias

        lp_init = log_prec_init / log_prec_lr
        self.log_prec_lr = log_prec_lr

        #if neuron_prec:
        self.u = nn.Parameter(B.randn(self.out_features, inducing_batch, 1))
        self.Xi = nn.Parameter(B.randn(1, inducing_batch, self.in_features+bias))

        precs = self.out_features if neuron_prec else 1
        self.log_prec_scaled = nn.Parameter(lp_init*t.ones(precs, 1, inducing_batch))
        if full_prec:
            self.prec_L = nn.Parameter(t.eye(inducing_batch).expand(precs, -1, -1))
        else:
            self.prec_L = None

        self._sample = None

    def forward(self, Xi):
        # inducing inputs are those stored, but expanded in first dimension to match inputs
        Xi = self.Xi.expand(Xi.shape[0], *self.Xi.shape)
        return rsample_logpq_weights_fc(self, Xi, neuron_prec=True)


class GIConv2dWeights(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, key=None, prior=NealPrior, stride=1, padding=0, inducing_targets=None, log_prec_init=-4., log_prec_lr=1., neuron_prec=False, inducing_batch=None):
        super().__init__()
        assert inducing_batch is not None
        assert inducing_batch != 0
        self.inducing_batch = inducing_batch

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        in_shape   = t.Size([in_channels, kernel_size, kernel_size])
        self.prior = prior(in_shape)

        self.stride = stride
        self.padding = padding
        self.log_prec_init = log_prec_init
        self.log_prec_lr = log_prec_lr
        self.neuron_prec = neuron_prec

        self.key = key

        self.u = None if (inducing_targets is None) else nn.Parameter(inducing_targets.clone().to(t.float64))

        lp_init = self.log_prec_init / self.log_prec_lr
        self.log_prec_scaled = nn.Parameter(lp_init*t.ones(self.inducing_batch))

        self._sample = None

    def forward(self, X):
        Xi = X[:, :self.inducing_batch, :, :, :]
        if self.u is None:
            (_, _, _, Hin, Win) = Xi.shape
            #HW_in = (Hin, Win)
            #HW_out = [(HW_in[i] + 2*self.padding - self.kernel_size) // self.stride + 1 for i in range(2)]
            self.u = nn.Parameter(B.randn(self.inducing_batch, self.out_channels, Hin, Win))

        sqrt_prec = (0.5 * self.log_prec_lr * self.log_prec_scaled).exp()[:, None, None, None]
        Xil = sqrt_prec * Xi
        Yil = sqrt_prec * self.u

        XiLXi, XiLY = conv_mm(Xil, Yil, self.kernel_size)
        XiLXi = XiLXi.unsqueeze(1)
        XiLY = XiLY.transpose(-1, -2).unsqueeze(-1)
        return rsample_logpq_weights(self, XiLXi, XiLY, self.prior, neuron_prec=True)


class LIConv2dWeights(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, prior=NealPrior, stride=1, padding=0, log_prec_init=-4., log_prec_lr=1., neuron_prec=False, full_prec=False):
        super().__init__()
        assert 1==kernel_size%2
        assert padding == kernel_size//2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        in_shape   = t.Size([in_channels, kernel_size, kernel_size])
        self.prior = prior(in_shape)

        self.stride = stride
        self.padding = padding
        self.log_prec_init = log_prec_init
        self.log_prec_lr = log_prec_lr
        self.neuron_prec = neuron_prec

        in_features = in_channels*kernel_size**2

        inducing_batch = in_features

        lp_init = log_prec_init / log_prec_lr
        self.log_prec_lr = log_prec_lr

        self.u = nn.Parameter(B.randn(self.out_channels, inducing_batch, 1))
        self.Xi = nn.Parameter(B.randn(1, inducing_batch, in_features))

        precs = out_features if neuron_prec else 1
        self.log_prec_scaled = nn.Parameter(lp_init*t.ones(precs, 1, inducing_batch))
        if full_prec:
            self.prec_L = nn.Parameter(t.eye(inducing_batch).expand(precs, -1, -1))
        else:
            self.prec_L = None

        self._sample = None

    def forward(self, Xi):
        # inducing inputs are those stored, but expanded in first dimension to match inputs
        Xi = self.Xi.expand(Xi.shape[0], *self.Xi.shape)
        return rsample_logpq_weights_fc(self, Xi, neuron_prec=self.neuron_prec)
        


class GILinear(AbstractLinear):
    r"""
    IID Gaussian prior and factorised Gaussian posterior over the weights of a fully-connected layer.

    arg:
        - **in_features:** size of each input sample
        - **out_features:** size of each output sample

    compulsory kwargs:
        - **inducing_batch:** This module assumes that the first ``inducing_batch`` elements of the minibatch are inducing, and the rest are test/training inputs. Can be combined with InducingWrapper to simplify working with inducing inputs.

    optional kwargs:
        - **bias:** If set to ``False``, the layer will not learn an additive bias.  Default: ``True``
        - **prior:** Prior over neural network weights.  Defaults: ``NealPrior``.
        - **inducing_targets:** Initial value of the inducing targets (useful at the top-layer, but never necessary).  Default: ``None``.
        - **neuron_prec:** Use a different precision parameter for each hidden neuron?  Considerably increases computational cost for relatively small performance benefit.  Default: ``False``.
        - **log_prec_init:** Initial value of the precision parameters.  The default assumes that little data is available.  Default: ``-4``. 
        - **log_prec_lr:** Multiplier for the learning rate of the precision parameters.  Default: ``1``.

    Shape:
        - **Input:** ``(samples, mbatch, in_features)``
        - **Output:** ``(samples, mbatch, out_features)``

    Examples:

        >>> import torch
        >>> import bayesfunc as bf
        >>> m = bf.GILinear(20, 30, inducing_batch=20)
        >>> input = torch.randn(3, 128, 20)
        >>> output, _, _ = bf.propagate(m, input)
        >>> print(output.size())
        torch.Size([3, 128, 30])
    """
    Weights = GILinearWeights

class GILinearFullPrec(nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        self.weights = GILinearWeights((in_features+bias)*out_features, 1, bias=False, **kwargs)

    def forward(self, X):
        assert X.shape[-1] == self.in_features
        (S, _, _) = X.shape

        if self.bias:
            ones = t.ones((*X.shape[:-1], 1), device=X.device, dtype=X.dtype)
            X = t.cat([X, ones], -1)

        zeros = t.zeros_like(X)
        X_trans = t.cat([t.cat([X if i==j else zeros for i in range(self.out_features)], -1) for j in range(self.out_features)], -2)
        #print(X.shape)
        #print(X_trans.shape)

        W_trans = self.weights(X_trans)
        #print(W_trans.shape)
        W = W_trans.view(S, self.out_features, self.in_features+self.bias)
        #print(W.shape)
        assert S==W.shape[0]
        
        result = X@W.transpose(-1, -2)

        assert result.shape[-1] == self.out_features
        assert 3 == len(result.shape)
        return result


class GIConv2d(AbstractConv2d):
    r"""
    IID Gaussian prior and factorised Gaussian posterior over the weights of a 2D convolutional layer.

    arg:
        - **in_channels:** number of channels in input tensor
        - **out_channels:** number of channels in output tensor
        - **kernel_size:** size of convolutional kernel

    compulsory kwargs:
        - **inducing_batch:** This module assumes that the first ``inducing_batch`` elements of the minibatch are inducing, and the rest are test/training inputs. Can be combined with InducingWrapper to simplify working with inducing inputs.

    optional kwargs:
        - **stride:** Standard convolutional stride.  Defaults to 1.
        - **padding:** Standard convolutional padding.  Defaults to 0.
        - **prior:** Prior over neural network weights.  Defaults: ``NealPrior``.
        - **inducing_targets:** Initial value of the inducing targets (useful at the top-layer, but never necessary).  Default: ``None``.
        - **neuron_prec:** Use a different precision parameter for each hidden neuron?  Considerably increases computational cost for relatively small performance benefit.  Default: ``False``.
        - **log_prec_init:** Initial value of the precision parameters.  The default assumes that little data is available.  Default: ``-4``. 
        - **log_prec_lr:** Multiplier for the learning rate of the precision parameters.  Default: ``1``.

    Shape:
        - **Input:** ``(samples, mbatch, in_height, in_width, in_features)``
        - **Output:** ``(samples, mbatch, in_height, in_width, out_features)``

    Warning:
        The inducing targets for this class are only initialised after a pass through the network (because it is only possible to infer the shape of the targets after it has seen an input).  As such, you must pass data through the network *before* calling ``opt(net.parameters(), lr=...)``.  Not doing so will silently cause poor performance.

    """
    Weights = GIConv2dWeights


class LILinear(AbstractLinear):
    r"""
    IID Gaussian prior and factorised Gaussian posterior over the weights of a fully-connected layer.
    ``inducing_batch`` is set to ``in_features+bias`` to give the smallest number of inducing points that is complete.

    arg:
        - **in_features:** size of each input sample
        - **out_features:** size of each output sample

    optional kwargs:
        - **bias:** If set to ``False``, the layer will not learn an additive bias.  Default: ``True``
        - **prior:** Prior over neural network weights.  Defaults: ``NealPrior``.
        - **neuron_prec:** Use a different precision parameter for each hidden neuron?  Considerably increases computational cost for relatively small performance benefit.  Default: ``False``.
        - **log_prec_init:** Initial value of the precision parameters.  The default assumes that little data is available.  Default: ``-4``. 
        - **log_prec_lr:** Multiplier for the learning rate of the precision parameters.  Default: ``1``.
        - **inducing_targets:** Initial value of the inducing targets.  Only useful in a single-layer net. Default: ``None``.
        - **inducing_batch:** Initial value of the inducing batch.  Only useful in a single-layer net.  Default: ``None``.

    Shape:
        - **Input:** ``(samples, mbatch, in_features)``
        - **Output:** ``(samples, mbatch, out_features)``

    Examples:

        >>> import torch
        >>> import bayesfunc as bf
        >>> m = bf.LILinear(20, 30)
        >>> input = torch.randn(3, 128, 20)
        >>> output, _, _ = bf.propagate(m, input)
        >>> print(output.size())
        torch.Size([3, 128, 30])
    """
    Weights = LILinearWeights


class LIConv2d(AbstractConv2d):
    r"""
    IID Gaussian prior and factorised Gaussian posterior over the weights of a 2D convolutional layer.
    ``inducing_batch`` is set to ``in_features+bias`` to give the smallest number of inducing points that is complete.

    arg:
        - **in_channels:** number of channels in input tensor
        - **out_channels:** number of channels in output tensor
        - **kernel_size:** size of convolutional kernel

    optional kwargs:
        - **stride:** Standard convolutional stride.  Defaults to 1.
        - **padding:** Standard convolutional padding.  Defaults to 0.
        - **prior:** Prior over neural network weights.  Defaults: ``NealPrior``.
        - **inducing_targets:** Initial value of the inducing targets (useful at the top-layer, but never necessary).  Default: ``None``.
        - **neuron_prec:** Use a different precision parameter for each hidden neuron?  Considerably increases computational cost for relatively small performance benefit.  Default: ``False``.
        - **log_prec_init:** Initial value of the precision parameters.  The default assumes that little data is available.  Default: ``-4``. 
        - **log_prec_lr:** Multiplier for the learning rate of the precision parameters.  Default: ``1``.

    Shape:
        - **Input:** ``(samples, mbatch, in_height, in_width, in_features)``
        - **Output:** ``(samples, mbatch, in_height, in_width, out_features)``

    """
    Weights = LIConv2dWeights

