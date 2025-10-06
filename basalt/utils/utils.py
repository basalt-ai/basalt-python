from typing import Dict, Any

def pick_typed(dict: Dict[str, Any], field_name: str, expected_type: Any) -> Any:
    value = dict.get(field_name)

    if not isinstance(value, expected_type):
        raise TypeError(f"Field '{field_name}' must be of type {expected_type.__name__}, got {type(value).__name__}.")

    # Additional check for int, because isinstance(True, int) == True
    if expected_type == int and isinstance(value, bool):
        raise TypeError(f"Field '{field_name}' must be of type int, got bool.")

    return value

def pick_number(dict: Dict[str, Any], field_name: str) -> float:
    value = dict.get(field_name)

    if isinstance(value, float):
        return float(value)

    # Additional check for int, because isinstance(True, int) == True
    if isinstance(value, bool) == False and isinstance(value, int):
        return int(value)

    raise TypeError(f"Field '{field_name}' must be a number (int or float), got {type(value).__name__}.")
