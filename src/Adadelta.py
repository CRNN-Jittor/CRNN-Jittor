import jittor as jt
from jittor import optim

class Adadelta(optim.Optimizer):
    """ Adadelta Optimizer.
    # Reference: https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
    Args:
        params(list): parameters of model.
        lr(float): learning rate.
        rho(float): coefficient used for computing a running average of squared gradients.
        eps(float): term added to the denomizator to avoid division by zero.
        weight_decay(float): weight decay (L2 penalty)

    Example::
        optimizer = Adadelta(model.parameters(), lr=0.001, rho=0.9, eps=1e-06, weight_decay=0)
        optimizer.step(loss)
    """
    def __init__(self, params, lr=0.001, rho=0.9, eps=1e-06, weight_decay=0):
        super().__init__(params, lr)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

        # initialize required arguments
        for pg in self.param_groups:
            v = pg["v"] = []
            u = pg["u"] = []
            for p in pg["params"]:
                v.append(jt.zeros(p.shape, p.dtype).stop_grad())
                u.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        v = group["v"] = []
        u = group["u"] = []
        for p in group["params"]:
            v.append(jt.zeros(p.shape, p.dtype).stop_grad())
            u.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None):
        if loss is not None:
            self.pre_step(loss)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            rho = pg.get("rho", self.rho)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            for p, g, v, u in zip(pg["params"], pg["grads"], pg["v"], pg["u"]):
                if p.is_stop_grad(): continue
                g = g + weight_decay * p
                v.update(rho * v + (1 - rho) * g * g)
                delta = jt.sqrt(u + eps) / jt.sqrt(v + eps) * g
                u.update(rho * u + (1 - rho) * delta**2)
                p.update(p - lr * delta)
        self.zero_grad()

# test
if __name__ == "__main__":
    import numpy as np
    from jittor import nn, Module, init

    ### model define
    class Model(Module):
        def __init__(self):
            self.layer1 = nn.Linear(1, 10)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(10, 1)
        def execute (self,x) :
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    np.random.seed(0)
    jt.set_seed(3)
    n = 1000
    batch_size = 50
    base_lr = 0.05
    # we need to stop grad of global value to prevent memory leak
    lr = jt.float32(base_lr).name("lr").stop_grad()

    def get_data(n):
        for i in range(n):
            x = np.random.rand(batch_size, 1)
            y = x*x
            yield jt.float32(x), jt.float32(y)

    model = Model()
    learning_rate = 0.1
    optim = Adadelta(model.parameters(), lr=0.1, rho=0.9, eps=1e-06, weight_decay=0)

    for i,(x,y) in enumerate(get_data(n)):
        pred_y = model(x)
        loss = jt.sqr(pred_y - y)
        loss_mean = loss.mean()
        optim.step (loss_mean)
        print(f"step {i}, loss = {loss_mean.data.sum()}")

    assert loss_mean.data < 0.005