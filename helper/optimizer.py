from theano import shared
import theano as theano
import numpy as np
import theano.tensor as T
from theano.compat.python2x import OrderedDict


dtype = T.config.floatX

class RMSprop:
    def __init__(self, cost, params, lr=0.0001, rho=0.9, epsilon=1e-6):

        self.cost = cost
        self.params = params
        self.lr = shared(np.cast[dtype](lr))
        self.rho = shared(np.cast[dtype](rho))
        self.epsilon = shared(np.cast[dtype](epsilon))
        self.gparams = T.grad(self.cost, self.params)

    def getUpdates(self):

        acc = [shared(np.zeros(p.get_value(borrow=True).shape, dtype=dtype)) for p in self.params]
        updates = []

        for p, g, a in zip(self.params, self.gparams, acc):
            new_a = self.rho * a + (1 - self.rho) * (g ** 2)
            updates.append((a, new_a))
            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))

        return updates
#
# class Adam():
#     def __init__(self,cost,params, lr=0.0001,alpha=0.001):
#         self.alpha = theano.shared(np.array(alpha).astype(theano.config.floatX))
#         self.cost = cost
#         self.params = params
#         self.lr = shared(np.cast[dtype](lr))
#
#     def getUpdates(self):
#         t = theano.shared(np.array(1).astype(theano.config.floatX))
#         alpha = self.alpha
#         beta_1 = np.array(0.9).astype(theano.config.floatX)
#         beta_2 = np.array(0.999).astype(theano.config.floatX)
#         epsilon = np.array(1.0*10**-8.0).astype(theano.config.floatX)
#         lam = np.array(1.0-1.0*10**-8.0).astype(theano.config.floatX)
#         g_model_params = []
#         models_m = []
#         models_v = []
#         updates = []
#         for param in self.params:
#             gparam = T.grad(self.cost, wrt=param)
#             g_model_params.append(gparam)
#             m = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX))
#             v = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX))
#             models_m.append(m)
#             models_v.append(v)
#         for param, gparam, m, v in zip(self.params, g_model_params, models_m, models_v):
#             beta_1_t = T.cast(beta_1 * lam ** (t - 1), theano.config.floatX)
#             m_new= T.cast(beta_1_t * m + (1 - beta_1_t) * gparam,theano.config.floatX)
#             v_new=T.cast(beta_2 * v + (1 - beta_2) * (gparam * gparam),theano.config.floatX)
#
#             #updates[m] = T.cast(beta_1_t * m + (1 - beta_1_t) * gparam,theano.config.floatX)
#             #updates[v] = T.cast(beta_2 * v + (1 - beta_2) * (gparam * gparam),theano.config.floatX)
#             m_hat = T.cast(m_new / (1 - beta_1 ** t), theano.config.floatX)
#             v_hat = T.cast(v_new / (1 - beta_2 ** t), theano.config.floatX)
#             p_new= param - alpha * m_hat / (T.sqrt(v_hat) + epsilon)
#             updates.append((m, m_new))
#             updates.append((v, v_new))
#             updates.append((param, p_new))
#         return updates

class Adam():
    def __init__(self,cost,params,lr=0.0001, b1=0.1, b2=0.001, e=1e-8):
        self.cost = cost
        self.params = params
        self.i = theano.shared(np.float32(0.))
        self.i_t = self.i + 1.
        fix1 = 1. - (1. - b1)**self.i_t
        fix2 = 1. - (1. - b2)**self.i_t
        self.lr = lr * (T.sqrt(fix2) / fix1)
        self.b1 = b1
        self.b2 = b2
        self.e=e
        self.gparams = T.grad(self.cost, self.params)
         # Gradient clipping
        clip_lower_bound=-1
        clip_upper_bound=1

        r_params=self.gparams[0:17]
        r_params =[T.clip(g, clip_lower_bound, clip_upper_bound) for g in r_params]
        self.gparams[0:17]=r_params

    def getUpdates(self):
        updates = []
        for p, g in zip(self.params, self.gparams):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (self.b1 * g) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(g)) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.e)
            p_t = p - (self.lr * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((self.i, self.i_t))
        return updates




class ClipRMSprop:
    def __init__(self, cost, params, lr=0.0001, rho=0.9, epsilon=1e-6, momentum=0.9,rescale=5.):
        self.cost = cost
        self.params = params
        self.momentum = momentum
        self.rescale=rescale
        self.lr = shared(np.cast[dtype](lr))
        self.rho = shared(np.cast[dtype](rho))
        self.epsilon = shared(np.cast[dtype](epsilon))
        self.gparams = T.grad(self.cost, self.params)

    def getUpdates(self):
        params=self.params
        lr=self.lr
        momentum=self.momentum
        rescale=self.rescale
        gparams =self.gparams
        updates = OrderedDict()

        if not hasattr(self, "running_average_"):
            self.running_square_ = [0.] * len(gparams)
            self.running_avg_ = [0.] * len(gparams)
            self.updates_storage_ = [0.] * len(gparams)

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        # Gradient clipping
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        for n, (param, gparam) in enumerate(zip(params, gparams)):
            gparam = T.switch(not_finite, 0.1 * param,
                              gparam * (scaling_num / scaling_den))
            combination_coeff = 0.9
            minimum_grad = 1e-4
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(gparam)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * gparam
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - lr * (
                gparam / rms_grad)
            self.running_square_[n] = new_square
            self.running_avg_[n] = new_avg
            self.updates_storage_[n] = update_step
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step

        return updates