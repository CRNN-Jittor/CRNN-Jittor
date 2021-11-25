from jittor import Var
from numpy import ndarray


def not_real(num):
    if isinstance(num, Var) or isinstance(num, ndarray):
        num = num.item()
    return num != num or num == float("inf") or num == float("-inf")
