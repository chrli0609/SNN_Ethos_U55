


def float_to_int_safe(x: float) -> int:
    if not x.is_integer():
        raise ValueError(f"Cannot convert {x} to int: it has nonzero decimals.")
    return int(x)