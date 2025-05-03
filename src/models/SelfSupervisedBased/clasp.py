import numpy as np
from typing import Optional
from src.models.ModelConstructors import ChangePointDetectionConstructor

class ClaSP(ChangePointDetectionConstructor):
    """ Basic class to work with ChangePoint detection models.

    Attributes:
        parameters: dict of parameters for selected model.

    """
    def __init__(self,
                 **kwargs):
        """ Highly used parameters.

        Args:

        """
        super().__init__(**kwargs)
        ...
