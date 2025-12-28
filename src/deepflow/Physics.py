import torch
from .Utility import calc_grad, calc_grads

class PDE():
    def __init__(self):
        pass

    def calc_residual_field(self):
        residuals_field = 0
        for residual_field in self.residual_fields:
            residuals_field += torch.abs(residual_field)
        return residuals_field

    def cal_loss_field(self):
        loss_field = 0
        for residual_field in self.residual_fields:
            loss_field += residual_field**2
        return loss_field

    def calc_loss(self):
        loss_field = 0
        for residual_field in self.residual_fields:
            loss_field += residual_field**2
        loss = torch.mean(loss_field)
        return loss
class NVS_nondimensional(PDE):
    def __init__(self, U, L, mu, rho):
        super().__init__()
        self.U = U
        self.L = L
        self.mu = mu
        self.rho = rho
        self.Re = rho*U*L/mu
        self.var = {}

    def calc(self, inputs_dict):
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']
        v = inputs_dict['v']
        p = inputs_dict['p']

        """
        Calculates the residuals of the incompressible Navier-Stokes equations.
        """
        # First-order derivatives (fewer autograd calls)
        if t is None:
            (u_x, u_y) = calc_grads(u, (x, y))
            (v_x, v_y) = calc_grads(v, (x, y))
            (p_x, p_y) = calc_grads(p, (x, y))
            u_t = None
            v_t = None
        else:
            (u_x, u_y, u_t) = calc_grads(u, (x, y, t))
            (v_x, v_y, v_t) = calc_grads(v, (x, y, t))
            (p_x, p_y) = calc_grads(p, (x, y))

        # Second-order derivatives
        u_xx= calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)
        v_xx = calc_grad(v_x, x)
        v_yy = calc_grad(v_y, y)
        
        # PDE residuals
        # Continuity equation (mass conservation)
        mass_residual = u_x + v_y
          
        if t is None:
            # X-momentum equation
            x_momentum_residual = (u * u_x + v * u_y -
                                (u_xx + u_yy)/self.Re +
                                p_x)
                                
            # Y-momentum equation
            y_momentum_residual = (u * v_x + v * v_y -
                                (v_xx + v_yy)/self.Re +
                                p_y)

        else:
            # X-momentum equation
            x_momentum_residual = (u_t + u * u_x + v * u_y -
                                (u_xx + u_yy)/self.Re +
                                p_x)
                                
            # Y-momentum equation
            y_momentum_residual = (v_t + u * v_x + v * v_y -
                                (v_xx + v_yy)/self.Re +
                                p_y)    

        self.residual_fields = (mass_residual, x_momentum_residual, y_momentum_residual)
        return self.residual_fields

    def dimensionalize(self):
        self.var['x'] /= self.L
        self.var['y'] /= self.L
        self.var['u'] /= self.U
        self.var['v'] /= self.U
        self.var['p'] /= (self.rho*self.U**2)
        if self.var['t']:
            self.var['t'] *= self.U/self.L

    def non_dimensionalize(self):
        self.var['x'] *= self.L
        self.var['y'] *= self.L
        self.var['u'] *= self.U
        self.var['v'] *= self.U
        self.var['p'] *= (self.rho*self.U**2)
        if self.var['t']:
            self.var['t'] /= (self.U/self.L)
class Heat(PDE):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.var = {}

    def calc_residual(self, inputs_dict):
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']

        # Derivatives (compute first-order in one call)
        (u_x, u_y, u_t) = calc_grads(u, (x, y, t))
        (u_xx, _) = calc_grads(u_x, (x, y))
        (_, u_yy) = calc_grads(u_y, (x, y))

        # Heat equation residual
        heat_residual = u_t - self.alpha * (u_xx + u_yy)

        self.residuals = (heat_residual,)
        return self.residuals