
class FormInteger:
    data_type = "integer"
    
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def __str__(self):
        return f"FormInteger(minimum {self.minimum}, maximum {self.maximum})"

    def is_valid(self, var):
        if isinstance(var, int):
            return (self.minimum is None or var >= self.minimum) and (self.maximum is None or var <= self.maximum)
        return False