import torch

class config:

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    TAG_CONSTANT = 1.0