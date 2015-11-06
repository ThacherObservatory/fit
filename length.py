import numpy as np

def length(something):
    if hasattr(something, "__len__"):
        try:
            return len(something)
        except:
            return len(np.atleast_1d(something))
    else:
        return 1
