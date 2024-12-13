
class FormString:
    data_type = "string"
    
    def __init__(self):
        pass

    def __str__(self):
        return "FormString"

    def is_valid(self, var):
        return isinstance(var, str)