# src/predicates_runtime.py
import math

def is_prime(x: int) -> bool:
    if x < 2: return False
    if x % 2 == 0: return x == 2
    r = int(math.isqrt(x))
    for p in range(3, r+1, 2):
        if x % p == 0: return False
    return True

def is_cube(x: int) -> bool:
    if x < 0:
        r = round(abs(x) ** (1/3))
        return -r**3 == x
    r = round(x ** (1/3))
    return r**3 == x

def is_square(x: int) -> bool:
    if x < 0:
        return False
    r = math.isqrt(x)
    return r * r == x