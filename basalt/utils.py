from re import sub as reg_replace
from typing import Tuple, Set, Dict, Any

def replace_variables(template: str, replacements: dict) -> Tuple[Set[str], str]:
    missing_keys = set([])

    count = lambda key: missing_keys.add(key)
    get = lambda key: replacements.get(key) if replacements.get(key) else count(key)
    replacer = lambda match: str(get(match.group(1))) if get(match.group(1)) else match.group(0)

    replaced = reg_replace(
        r'{{(.*?)}}',
        replacer,
        template
    )

    return missing_keys, replaced


def pick_typed(dict: Dict[str, Any], field_name: str, expected_type: Any) -> Any:
    value = dict.get(field_name)

    if not isinstance(value, expected_type):
        raise TypeError(f"Field '{field_name}' must be of type {expected_type.__name__}, got {type(value).__name__}.")

    return value
