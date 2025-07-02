class DataframeMissingColumnsError(Exception):
    """Raised when the dataframe is missing specific columns for the desired action"""

    def __init__(self, msg):
        super().__init__(msg)


class ApiNotStarted(Exception):
    """
    Raised when the required API is not started

    Args:
        Exception ([type]): Inherits from Exception
    """

    def __init__(self, msg):
        super().__init__(msg)
