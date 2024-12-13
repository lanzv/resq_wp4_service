from datetime import datetime

class FormTime:
    data_type = "time"
    
    def __init__(self):
        pass

    def __str__(self):
        return "FormTime"

    def is_valid(self, var):
        if not isinstance(var, str):
            return False

        formats = ["%H:%M:%S"]
        for fmt in formats:
            try:
                datetime.strptime(var, fmt)
                return True
            except ValueError:
                continue

        return False