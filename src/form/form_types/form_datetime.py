from datetime import datetime

class FormDateTime:
    data_type = "date-time"
    
    def __init__(self):
        pass

    def __str__(self):
        return "FormDateTime"

    def is_valid(self, var):
        if not isinstance(var, str):
            return False

        formats = [
            "%Y-%m-%dT%H:%M:%S"
        ]

        for fmt in formats:
            try:
                datetime.strptime(var, fmt)
                return True
            except ValueError:
                continue

        return False