class Fps(torch.nn.Module):
    def __init__(self):
        super(Fps, self).__init__()
    
    def forward(self,data):
        sampled_data = data
        return data