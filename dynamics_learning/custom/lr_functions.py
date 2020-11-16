def lr1(step: int, base_lr: float) -> float:
    """Medium aggression lr scheduler."""
    lr = base_lr
    _lr = lr * 0.975 ** (step // 20)
    return max(_lr, lr * 1e-3)  # default


def lr2(step: int, base_lr: float) -> float:
    """Stronger aggression lr scheduler.

    DO NOT EDIT. GOOD SETTING FOR KVAE+PEND_IMG.
    """
    lr = base_lr
    _lr = lr * 0.975 ** (step // 100)
    return max(_lr, lr * 1e-2)  # default


def lr3(step: int, base_lr: float) -> float:
    """Stronger aggression lr scheduler."""
    lr = base_lr
    _lr = lr * 0.9 ** (step // 25)
    return max(_lr, lr * 1e-3)  # default


def lr4(step: int, base_lr: float) -> float:
    """Warmup for pre-trained image model."""
    lr = base_lr
    warmup_period = 1000
    if step < warmup_period:
        _lr = ((step + 1) / warmup_period) * lr
    else:
        _lr = lr2(step - warmup_period, lr)
    return _lr


def lr5(step: int, base_lr: float) -> float:
    """LR5."""
    lr = base_lr
    _lr = lr * 0.9 ** (step // 100)
    return max(_lr, 1e-6)  # default
