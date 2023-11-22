""" Atmospheric MicroPhysics (amp) class 
"""

__all__ = ['AmpCldAM']


class AmpCloud():
    """Common Amp cloud
    """
    def __init__(self):
        self.cloudmodel = None # cloud model
        

class AmpAmcloud(AmpCloud):
    def __init__(self, pdb):
        """initialization of amp for Ackerman and Marley 2001 cloud model

        Args:
            pdb (_type_): particulates database (pdb) 
        """
        self.cloudmodel = "Ackerman and Marley (2001)"


