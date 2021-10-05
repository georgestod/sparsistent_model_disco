#############################################################################
# DeepModx, a not so beautiful JAX version of DeepMod, yet working code
# Authors: Gert-Jan Both (initial version, January 2021) and Georges Tod (January 2021 - today)
#############################################################################

from typing import (Any, Sequence, Callable, Iterable, Optional, Tuple, Union)
from functools import partial
from dataclasses import dataclass

import jax
from jax import lax, random, vmap, numpy as jnp
from jax._src.nn.initializers import _compute_fans
from jax import value_and_grad, jit
from jax.ops import index_update, index

from flax import linen as nn
from flax.core import freeze, unfreeze
import flax.struct

from tensorboardX import SummaryWriter


#############################################################################
####### Training
#############################################################################

def loss_fn_pinn_multi2(params, state, model, x, y):
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, coeffs_mask), updated_state = model.apply(variables, x, mutable=list(state.keys()))
    MSE = jnp.sum(mse(prediction, y))
    Reg = jnp.sum(mse(dt.squeeze().T, (theta @ coeffs).squeeze().T))
    loss = MSE + Reg
    metrics = {"loss": loss, "mse": MSE, "reg": Reg, "coeff": coeffs, "mask": coeffs_mask}
    return loss, (updated_state, metrics, dt, theta, coeffs, coeffs_mask)

def loss_data_only(params, state, model, x, y):
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, coeffs_mask), updated_state = model.apply(variables, x, mutable=list(state.keys()))
    MSE = jnp.sum(mse(prediction, y))
    Reg = jnp.sum(mse(dt.squeeze().T, (theta @ coeffs).squeeze().T))
    loss = MSE
    metrics = {"loss": loss, "mse": MSE, "reg": Reg, "coeff": coeffs, "mask": coeffs_mask}
    return loss, (updated_state, metrics, dt, theta, coeffs, coeffs_mask)

def create_update_state(loss_fn, *args, **kwargs):
    # Function to create fast update for flax models with variables
    def step(opt, state, loss_fn, *args, **kwargs):
        grad_fn = value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, (updated_state, metrics, dt, theta, coeffs, coeffs_mask)), grad = grad_fn(opt.target, state, *args, **kwargs)
        opt = opt.apply_gradient(grad)  # Return the updated optimizer with parameters.
        return (opt, updated_state), metrics, dt, theta, coeffs, coeffs_mask

    return jit(lambda opt, state: step(opt, state, loss_fn, *args, **kwargs))

@dataclass
class Convergence:
    patience = 5000
    delta: float = 1e-2
    start_iteration = None
    start_norm = None

    def __call__(self, iteration, coeffs):
        coeff_norm = jnp.linalg.norm(coeffs)

        # Initialize if doesn't exist
        if self.start_norm is None:
            self.start_norm = coeff_norm
            self.start_iteration = iteration
            converged = False

        # Check if change is smaller than delta and if we've exceeded patience
        elif jnp.abs(self.start_norm - coeff_norm).item() < self.delta:
            if (iteration - self.start_iteration) >= self.patience:
                converged = True
            else:
                converged = False

        # If not, reset and keep going
        else:
            self.start_norm = coeff_norm
            self.start_iteration = iteration
            converged = False

        return converged
    
@dataclass
class mask_scheduler:
    patience: int = 2000
    delta: float = 1e-6
    periodicity: int = 1000
    periodic: bool = False
    best_loss = None
    best_iteration = None
    
    def __call__(self, loss, iteration, optimizer):
        if self.periodic is True:
            if (iteration - self.best_iteration) % self.periodicity == 0:
                update_mask, optimizer = True, optimizer
            else:
                 update_mask, optimizer = False, optimizer
        elif self.best_loss is None:
            self.best_loss = loss
            self.best_iteration = iteration
            self.best_optim_state = optimizer
            update_mask, optimizer = False, optimizer
        # If it didnt improve, check if we're past patience
        elif (self.best_loss - loss) < self.delta:

            if (iteration - self.best_iteration) >= self.patience:
                self.periodic = True  # switch to periodic regime
                self.best_iteration = iteration  # because the iterator doesnt reset
                update_mask, optimizer = True, self.best_optim_state
            else:
                update_mask, optimizer = False, optimizer
        # If not, keep going
        else:
            self.best_loss = loss
            self.best_iteration = iteration
            self.best_optim_state = optimizer
            update_mask, optimizer = False, optimizer
        return update_mask, optimizer
    
    
#############################################################################    
####### Model
#############################################################################

class pdeX(nn.Module): 
    shared_features: Sequence[int]
    specific_features: Sequence[int]
    n_tasks: int

    @nn.compact
    def __call__(self, inputs):

        # computing time derivative and library
        u_field_observed = MultiTaskMLPSiren(self.shared_features, self.specific_features, self.n_tasks)
        prediction, dt, theta = library_5th(u_field_observed,inputs,)
        
        
        # checking if some variable is initialized
        is_initialized = self.has_variable("coeffs_mask", "maskC")
        coeffs_mask_d = self.variable("coeffs_mask", "maskC", lambda s: jnp.ones(s,dtype=jnp.uint8), [theta.shape[0],theta.shape[2],1])
        coeffs_mask = coeffs_mask_d.value 
        
        # normalizing library
        nt = jnp.linalg.norm(theta,axis=1,keepdims=True)
        nts = nt.reshape(coeffs_mask.shape)
        normed_theta = theta/nt

        masked_thetas = jax.vmap(jnp.multiply,in_axes=0,out_axes=0)(normed_theta,coeffs_mask[:,:,0]) # masking inactive library terms
        coeffs        = RidgeMT()((masked_thetas, dt))/nts                                           # computing coeffs    
        masked_coeffs = coeffs*coeffs_mask                                                           # masking inactive residual coefficients
        
        
        return prediction, dt, theta, masked_coeffs, coeffs_mask
    
#############################################################################    
####### Neural nets
#############################################################################

# source: Sirens (https://arxiv.org/pdf/2006.09661.pdf)

class MultiTaskMLPSiren(nn.Module):
    shared_features: Sequence[int]
    specific_features: Sequence[int]
    n_tasks: int
    omega_0: int=30
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs  # we overwrite x so we copy it to a new tensor
        idx1_ = -1
        for idx1_, feature in enumerate(self.shared_features):
            if idx1_==0:
                x = jnp.sin(self.omega_0 * DenseSiren0(feature)(x))
            else:
                x = jnp.sin(self.omega_0 * DenseSiren(feature)(x))                
        x = jnp.repeat(jnp.expand_dims(x, axis=0), repeats=self.n_tasks, axis=0)
        for idx2_, feature in enumerate(self.specific_features[:-1]):
            idx2_ = idx2_+idx1_
            if idx2_==-1:
                x = jnp.sin(self.omega_0 * MultiTaskDenseSiren0(feature, self.n_tasks)(x))
            else:
                x = jnp.sin(self.omega_0 * MultiTaskDenseSiren(feature, self.n_tasks)(x))        
        x = MultiTaskDenseSiren(self.specific_features[-1], self.n_tasks)(x)
        
        return (x.T)[0]


PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any
# to initialze kernels of both Dense and MultitaskDense layers
def siren_init0(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        #fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        fan_in = shape[-2]
        a = 1 / fan_in
        return jax.random.uniform(key, shape, dtype, minval=-a, maxval=a)
    return init
def siren_init(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        #fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        fan_in = shape[-2]
        a = jnp.sqrt(6 / fan_in) / 30
        return jax.random.uniform(key, shape, dtype, minval=-a, maxval=a)
    return init

# to initialze biases of Dense Layers
def siren_init_bias0(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        fan_in = shape[0]
        a = 1 / fan_in
        return jax.random.uniform(key, shape, dtype, minval=-a, maxval=a)
    return init
def siren_init_bias(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        fan_in = shape[0]
        a = jnp.sqrt(6 / fan_in) / 30
        return jax.random.uniform(key, shape, dtype, minval=-a, maxval=a)
    return init

class DenseSiren0(nn.Module):
    features: int    
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = siren_init0()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = siren_init_bias0()
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',self.kernel_init,(inputs.shape[-1], self.features))
        y = lax.dot_general(inputs, kernel,(((inputs.ndim - 1,), (0,)), ((), ())))
        bias = self.param('bias', self.bias_init, (self.features,))     
        y = y + bias
        return y
    
class DenseSiren(nn.Module):
    features: int    
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = siren_init()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = siren_init_bias()
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',self.kernel_init,(inputs.shape[-1], self.features))
        y = lax.dot_general(inputs, kernel,(((inputs.ndim - 1,), (0,)), ((), ())))
        bias = self.param('bias', self.bias_init, (self.features,))
        y = y + bias
        return y

class MultiTaskDenseSiren0(nn.Module):
    features: int
    n_tasks: int
    kernel_init: Callable = siren_init0()
    bias_init: Callable = siren_init0()
        
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param("kernel", self.kernel_init,(self.n_tasks, inputs.shape[-1], self.features))
        y = lax.dot_general(inputs, kernel, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
        bias = self.param("bias", self.bias_init, (self.n_tasks, 1, self.features))
        y = y + bias
        return y
    
class MultiTaskDenseSiren(nn.Module):
    features: int
    n_tasks: int
    kernel_init: Callable = siren_init()
    bias_init: Callable = siren_init()
        
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param("kernel", self.kernel_init,(self.n_tasks, inputs.shape[-1], self.features))
        y = lax.dot_general(inputs, kernel, dimension_numbers=(((2,), (1,)), ((0,), (0,))))
        bias = self.param("bias", self.bias_init, (self.n_tasks, 1, self.features))
        y = y + bias
        return y

    
#############################################################################    
####### A linear (ridge) regression to constraint the NN - aka PINN
#############################################################################
class RidgeMT(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        theta, dt = inputs
        coeffs =  self.myRidge(theta, dt) 
        return coeffs
    
    def myRidge(self, X, y):
        def fridge(X,y):
            lambdaC = 1e-4
            M1 = (X.T @ X + lambdaC*jnp.eye(X.shape[1])) 
            M2 =  X.T @ y
            cR = jnp.linalg.inv(M1) @ M2
            return cR

        xi_myRidge = jax.vmap(fridge, in_axes=(0, 0), out_axes=0)(X,y)
        return xi_myRidge

class Ridge(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        theta, dt = inputs
        coeffs =  self.fridge(theta, dt) 
        return coeffs
    

    def fridge(self,X,y):
        lambdaC = 1e-4
        M1 = (X.T @ X + lambdaC*jnp.eye(X.shape[1])) 
        M2 =  X.T @ y
        cR = jnp.linalg.inv(M1) @ M2
        return cR


        xi_myRidge = fridge(X,y)
    
        return xi_myRidge
#############################################################################    
####### Libraries
#############################################################################

def vgrad_backward(f, x):
    y, vjp_fn = jax.vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]

def vgrad_forward(f, x, input_idx):
    s = index_update(jnp.zeros_like(x), index[:, input_idx], 1)
    _, jvp = jax.jvp(f, (x,), (s,))
    return jvp

def vgrad_backward_vmap(f,x,M):
    y, vjp_fn = jax.vjp(f, x)
    outs, = vmap(vjp_fn)(M)
    return outs

def vgrad_forward_vmap(f, x, M):
    _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
    return vmap(_jvp)(M)


def library_5th(f, x):
    # Taking 5 first derivatives
    df = partial(vgrad_forward, f, input_idx=1)
    d2f = partial(vgrad_forward, df, input_idx=1)
    d3f = partial(vgrad_forward, d2f, input_idx=1)
    d4f = partial(vgrad_forward, d3f, input_idx=1)
    d5f = partial(vgrad_forward, d4f, input_idx=1)

    pred = jnp.expand_dims(f(x).T, axis=-1)
    dt = jnp.expand_dims(vgrad_forward(f, x, input_idx=0).T, axis=-1)

    # Polynomials up to 5th order
    u = jnp.concatenate([jnp.ones_like(pred), pred, pred ** 2, pred ** 3, pred ** 4, pred ** 5], axis=-1)
    du = jnp.concatenate(
        [
            jnp.ones_like(pred),
            jnp.expand_dims(df(x).T, axis=-1),
            jnp.expand_dims(d2f(x).T, axis=-1),
            jnp.expand_dims(d3f(x).T, axis=-1),
            jnp.expand_dims(d4f(x).T, axis=-1),
            jnp.expand_dims(d5f(x).T, axis=-1)

        ],axis=-1,)

    theta = (jnp.expand_dims(u, axis=-1) @ jnp.expand_dims(du, axis=-2)).reshape(u.shape[0], u.shape[1], 36)

    return (pred.T)[0], dt, theta

#############################################################################    
####### Losses
#############################################################################
def mse(y_pred, y):
    return jnp.mean((y_pred-y)**2, axis=0)

def mse_test(model, params, state, X, y):
    variables = {"params": params, **state}
    (prediction, dt, theta, coeffs, coeffs_mask), updated_state = model.apply(variables, X, mutable=list(state.keys()))
    loss_c   = mse(prediction, y)
    loss = jnp.sum(loss_c)
    return loss, loss_c

#############################################################################    
####### TensorboardX logger
#############################################################################    
class Logger:
    def __init__(self, path: str='writing_on_Mars', *args, **kwargs):
        self.writer = SummaryWriter(path,*args, **kwargs)

    def write(self, metrics, epoch):

        def rename_keys2(d):
            #to rename as strings
            return dict([(str(k), v) for k, v in d.items()])
        
        for key, value in metrics.items():

            # section 1
            if key=='coeff':
                if len(value.shape)<3:
                    name = '1_Coeffs/exp_'
                    d = dict(enumerate(value[:,0].flatten(), 1))
                    d = rename_keys2(d)                    
                    self.writer.add_scalars(name,d,epoch)
                else:
                    for exp in range(value.shape[0]):
                        name = '1_Coeffs/exp_' + str(exp)
                        d = dict(enumerate(value[exp,:,0].flatten(), 1))
                        d = rename_keys2(d)                    
                        self.writer.add_scalars(name,d,epoch)

            # section 2
            if key=='mask':
                value = value.astype(int)
                if len(value.shape)<3:

                    name = '2_Masks/exp_'
                    d = dict(enumerate(value[:,0].flatten(), 1))
                    d = rename_keys2(d)
                    self.writer.add_scalars(name,d,epoch)
                else:
                    for exp in range(value.shape[0]):
                        name = '2_Masks/exp_' + str(exp)
                        d = dict(enumerate(value[exp,:,0].flatten(), 1))
                        d = rename_keys2(d)
                        self.writer.add_scalars(name,d,epoch)
                        
            # section 3
            if key=='loss':
                self.writer.add_scalar('3_Training/'+key, value, epoch)
            if key=='mse':
                self.writer.add_scalar('3_Training/'+key, value, epoch)
            if key=='reg':
                self.writer.add_scalar('3_Training/'+key, value, epoch)  
                
            # section 4
            if key=='loss_test':
                d = dict(enumerate(value.flatten(), 1))
                d = rename_keys2(d)
                name = '4_Test/mse_test'
                self.writer.add_scalars(name,d,epoch)
                
            # section 5
            if key=='prob_selec':
                name = '5_Proba_Selected/probabilities'
                d = dict(enumerate(value, 1))
                d = rename_keys2(d)
                self.writer.add_scalars(name,d,epoch)

            # section 6
            if key=='cond':
                name = '6_cond/'
                d = dict(enumerate(value, 1))
                d = rename_keys2(d)
                self.writer.add_scalars(name,d,epoch)
                
            # section 7
            if key=='IRC':
                self.writer.add_scalar('7_IRC/'+key, value, epoch)
            if key=='IRC_Adaptive':
                self.writer.add_scalar('7_IRC/'+key, value, epoch)
            if key=='PoV':
                self.writer.add_scalar('7_IRC/'+key, value, epoch)
            if key=='cond_num_adaptive':
                self.writer.add_scalar('7_IRC/'+key, value, epoch)
            if key=='cond_num_adaptive_GT':
                self.writer.add_scalar('7_IRC/'+key, value, epoch)
                
                
    def close(self,):
        self.writer.flush()
        self.writer.close()