#--------
# courtesy of justin john
#--------
import random

import torch

import backend



def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-7):
    
    # fix random seed to 0
    backend.reset_seed(0)
    for i in range(num_checks):

        ix = tuple([random.randrange(m) for m in x.shape])

        oldval = x[ix].item()
        x[ix] = oldval + h  # increment by h
        fxph = f(x).item()  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x).item()  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error_top = abs(grad_numerical - grad_analytic)
        rel_error_bot = abs(grad_numerical) + abs(grad_analytic) + 1e-12
        rel_error = rel_error_top / rel_error_bot
        msg = "numerical: %f analytic: %f, relative error: %e"
        print(msg % (grad_numerical, grad_analytic, rel_error))


def compute_numeric_gradient(f, x, dLdf=None, h=1e-7):
 
    flat_x = x.contiguous().flatten()
    grad = torch.zeros_like(x)
    flat_grad = grad.flatten()

    # Initialize upstream gradient to be ones if not provide
    if dLdf is None:
        y = f(x)
        dLdf = torch.ones_like(y)
    dLdf = dLdf.flatten()

    # iterate over all indexes in x
    for i in range(flat_x.shape[0]):
        oldval = flat_x[i].item()  # Store the original value
        flat_x[i] = oldval + h  # Increment by h
        fxph = f(x).flatten()  # Evaluate f(x + h)
        flat_x[i] = oldval - h  # Decrement by h
        fxmh = f(x).flatten()  # Evaluate f(x - h)
        flat_x[i] = oldval  # Restore original value

        # compute the partial derivative with centered formula
        dfdxi = (fxph - fxmh) / (2 * h)

        # use chain rule to compute dLdx
        flat_grad[i] = dLdf.dot(dfdxi).item()

    # Note that since flat_grad was only a reference to grad,
    # we can just return the object in the shape of x by returning grad
    return grad


def rel_error(x, y, eps=1e-10):

    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot
