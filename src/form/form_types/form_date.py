from datetime import datetime

class FormDate:
    data_type = "date"
    
    def __init__(self):
        pass

    def __str__(self):
        return "FormDate"

    def is_valid(self, var):
        try:
            return isinstance(var, str) and datetime.strptime(var, "%Y-%m-%d")
        except ValueError:
            return False
