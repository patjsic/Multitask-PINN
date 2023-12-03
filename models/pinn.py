import torch
from torch import nn, autograd

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )[0]

class PINN(nn.Module):
    """
    Vanilla PINN with no maximal weight sharing, i.e. no split network heads
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu"):
        #Note: device should be set at training and fed into model call
        # default "cpu" is a fallback
        super(PINN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.criterion = nn.MSELoss(reduction="mean")
        self.activation = nn.Tanh()

        #Define loss variables for tensorboard logging
        self.data_loss_val = 0.0
        self.pde_loss_val = 0.0

        #Set default model weights to float
        self.float()

        #Create model layers
        self.mid_layers = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.Tanh())
        self.last_layer = nn.Linear(self.hidden_dim, self.output_dim)

        #Define lambda as learnable parameters
        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(0.01))

        #Initialize model weights
        self.init_weights()
    
    def init_weights(self):
        """
        For each nn.Linear layer in the model, randomly initialize weights and zero bias vectors
        """
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.constant_(mod.bias, 0.0)
    
    def compute_loss(self, u, v, u_pred, v_pred, f_pred, g_pred):
        """
        Calculate total loss and log for tensorboard
        """
        u_loss = self.criterion(u_pred, u)
        v_loss = self.criterion(v_pred, v)
        f_loss = self.criterion(f_pred, torch.zeros_like(f_pred))
        g_loss = self.criterion(g_pred, torch.zeros_like(g_pred))

        data_loss = u_loss + v_loss
        pde_loss = f_loss + g_loss

        self.data_loss_val = data_loss
        self.pde_loss_val = pde_loss
        return  data_loss + pde_loss

    def forward_ffn(self, input):
        """
        Forward pass for pure data input (no pde)
        """
        x = self.mid_layers(input)
        x = self.last_layer(x)
        return x
    
    def forward(self, x, y, t, p, u, v):
        """
        Calculate forward pass for model with pde loss
        """
        input = torch.stack([x, y, t], dim=1)
        #TODO: Figure out how to normalize input

        output = self.forward_ffn(input)
        s = output[:,0]
        p_pred = output[:,1]
        u_pred = grad(s, y)
        v_pred = -1 * grad(s, x)

        #Calculate gradients
        u_t = grad(u_pred ,t)
        v_t = grad(v_pred, t)
        u_x = grad(u_pred, x)
        v_x = grad(v_pred, x)
        p_x = grad(p_pred, x)
        u_y = grad(u_pred, y)
        v_y = grad(v_pred, y)
        p_y = grad(p_pred, y)
        u_xx = grad(u_x, x)
        v_xx = grad(v_x, x)
        u_yy = grad(u_y, y)
        v_yy = grad(v_y, y)

        #Calculate f and g
        f = self.lambda1 * (u_t + u_pred * u_x + v_pred * u_y) + p_x - self.lambda2 * (u_xx + u_yy)
        g = self.lambda1 * (v_t + u_pred * v_x + v_pred * v_y) + p_y - self.lambda2 * (v_xx + v_yy)

        return {"p_pred": p_pred, "u_pred": u_pred, "v_pred": v_pred, "f": f, "g": g}
        