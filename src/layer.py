class Layer:
    """
    Abstract class made to be a regular layer
    """
    def __init__(self):
        """
        Initializes an object of the Layer class
        """
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        """
        Will be implemented in other classes.
        """
        raise NotImplementedError

    def backward_propagation(self, output, learning_rate):
        """
        Will be implemented in other classes.
        """
        raise NotImplementedError
