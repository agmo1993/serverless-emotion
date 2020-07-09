import urllib.request
from PIL import Image


def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """
    url = req
    image = Image.open(urllib.request.urlopen(url))
    width, height = image.size
    

    return (width,height)
