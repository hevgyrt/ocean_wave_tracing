
class Advection():
    #def set_variables(self, cg, k, kx, U):
    def __init__(self, cg, k, kx, U):
        #cg, k, kx, U = self.cg, self.k, self.kx, self.U
        self.cg = cg
        self.k = k
        self.kx = kx
        self.U = U

    def __call__(self,u,t):
        cg, k, kx, U = self.cg, self.k, self.kx, self.U

        #f = [cg*(kx/k) + U]
        f = cg*(kx/k) + U
        return f

class WaveNumberEvolution():
    def __init__(self,d_sigma, kx, ky, dUkx, dUky):
        self.d_sigma = d_sigma
        self.kx = kx
        self.ky = ky
        self.dUkx = dUkx
        self.dUky = dUky

    def __call__(self, u, t):
        d_sigma, kx, ky, dUkx, dUky = self.d_sigma, self.kx, self.ky, self.dUkx, self.dUky

        #f = [d_sigma + kx*dUkx + ky*dUky]
        f = -(d_sigma + kx*dUkx + ky*dUky)
        return f


class ODESolver(object):
    """
    Based on the work by Langtangen, H.P. DOI = 10.1007/978-3-662-49887-3.
    Superclass for numerical methods solving scalar and vector ODEs
      du/dt = f(u, t)
    Attributes:
    t: array of time values
    u: array of solution values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(u, t)
    """
    def __init__(self, f):
        if not callable(f):
            raise TypeError('f is %s, not a function' % type(f))
        # For ODE systems, f will often return a list, but
        # arithmetic operations with f in numerical methods
        # require that f is an array. Let self.f be a function
        # that first calls f(u,t) and then ensures that the
        # result is an array of floats.
        self.f = lambda u, t: np.asarray(f(u, t), float)

    def advance(self):
        """Advance solution one time step."""
        raise NotImplementedError



class ForwardEuler(ODESolver):
    def advance(u, f, k, t):
        dt = t[k+1] - t[k]
        u_new = u + dt*f(u, t[k])
        return u_new

class RungeKutta4(ODESolver):
    def advance(u, f, k, t):
        dt = t[k+1] - t[k]
        dt2 = dt/2.0
        K1 = dt*f(u, t[k])
        K2 = dt*f(u + 0.5*K1, t[k] + dt2)
        K3 = dt*f(u + 0.5*K2, t[k] + dt2)
        K4 = dt*f(u + K3, t[k] + dt)
        u_new = u + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new
