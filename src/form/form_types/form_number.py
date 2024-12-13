
class FormNumber:
    data_type = "number"
    
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def __str__(self):
        return f"FormNumber(minimum {self.minimum}, maximum {self.maximum})"

    def is_valid(self, var):
        if isinstance(var, (float, int)):
            return (self.minimum is None or var >= self.minimum) and (self.maximum is None or var <= self.maximum)
        return False