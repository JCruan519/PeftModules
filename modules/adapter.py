from torch import nn

class Adapter(nn.Module):
    def __init__(self, dim, hidden_dim=4):
        super().__init__() 
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.mid_act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.scale_adapter = 0.1

    def forward(self, x):
        return (self.linear2(self.mid_act(self.linear1(x)))) * self.scale_adapter