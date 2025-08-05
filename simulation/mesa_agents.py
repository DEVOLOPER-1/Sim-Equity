class EnvironmentInitializer:
    """Initialize environment parameters for simulation.

    Parameters
    ----------
    center : tuple[float, float]
        The x,y coordinates of the environment center point.
    radius : float
        The radius of the environment boundary from the center point.
    time : str
        The simulation starts time in format 'month:day:hour:minute' (e.g., '08:05:14:30')
    """

    def __init__(
            self,
            center: tuple[float, float],
            radius:float,
            time:str
    )->None:
        self.radius = radius
        self.x = center[0]
        self.y = center[1]
        self.time = time
