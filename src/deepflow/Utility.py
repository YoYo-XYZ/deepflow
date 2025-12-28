import torch

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

def get_device():
    global device
    return device

def manual_seed(seed:int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)          # For current GPU
        torch.cuda.manual_seed_all(seed)      # For all GPUs

def calc_grad(y, x):
    """
    Calculates the gradient of a tensor y with respect to a tensor x.

    Returns:
        torch.Tensor: The gradient of y with respect to x.
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True)[0]
    return grad

def calc_grads(y_list, x_list):
    """
    Calculates the gradients of tensors y_list with respect to tensors x_list.

    Returns:
        tuple of torch.Tensor: The gradients of y_list with respect to x_list.
    """
    grads = torch.autograd.grad(
        outputs=y_list,
        inputs=x_list,
        grad_outputs=torch.ones_like(y_list),
        create_graph=True)
    return grads

def to_require_grad(*tensors):
    if len(tensors) == 1:
        return tensors[0].clone().detach().requires_grad_(True)
    else:
        return (t.clone().detach().requires_grad_(True) for t in tensors)

def torch_to_numpy(*tensors):
    def to_numpy(x):
        try:
            return x.detach().numpy()
        except:
            return x.numpy()

    if len(tensors) == 1:
        return to_numpy(tensors[0])
    else:
        return tuple(to_numpy(x) for x in tensors)