import torch

def cubicBezier(y0:torch.Tensor, yf:torch.Tensor, x:torch.Tensor):
    if torch.any(torch.logical_or(x < 0, x > 1)):
        print("x is out of range")
        exit()
    yDiff = yf - y0
    bezier = x * x * x + 3.0 * (x * x * (1.0 - x))
    return y0 + bezier[:, None] * yDiff

def cubicBezierFirstDerivative(y0:torch.Tensor, yf:torch.Tensor, x:torch.Tensor):
    if torch.any(torch.logical_or(x < 0, x > 1)):
        print("x is out of range")
        exit()
    yDiff = yf - y0
    bezier = 6.0 * x * (1.0 - x)
    return bezier[:, None] * yDiff

def cubicBezierSecondDerivative(y0:torch.Tensor, yf:torch.Tensor, x:torch.Tensor):
    if torch.any(torch.logical_or(x < 0, x > 1)):
        print("x is out of range")
        exit()
    yDiff = yf - y0
    bezier = 6.0 - 12.0 * x
    return bezier[:, None] * yDiff

