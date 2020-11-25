import torch

class Channel():
    def __init__(self, vocab_size, p=0.0):
        self.p = p
        self.vocab_size = vocab_size
    
    def __call__(self, message : torch.Tensor):
        repl_choice = torch.bernoulli(torch.full_like(message, self.p, dtype=torch.double)) == 1.0
        repl_value  = torch.randint_like(message, 1, self.vocab_size)

        inv_zero_mask = ~(message == 0)

        message = (
            message + inv_zero_mask * repl_choice * (repl_value - message)
        )

        return message

