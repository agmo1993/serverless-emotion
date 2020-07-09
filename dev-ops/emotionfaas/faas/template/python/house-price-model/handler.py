import numpy as np
def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    return {"Price" : np.random.randint(100000,500000,size=1)}
