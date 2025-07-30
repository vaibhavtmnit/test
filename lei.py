import re

def is_valid_lei(lei: str) -> bool:
    """
    Validates a single LEI string against the ISO 17442 standard.

    An LEI is valid if:
    1. It is 20 characters long.
    2. It contains only alphanumeric characters.
    3. It satisfies the ISO 7064 MOD 97-10 checksum algorithm.

    Args:
        lei: The potential LEI string to validate.

    Returns:
        True if the LEI is valid, False otherwise.
    """
    # 1. & 2. Check length and characters (already handled by regex but good for standalone use)
    if not re.fullmatch(r'[A-Z0-9]{20}', lei):
        return False

    # 3. Verify the checksum (MOD 97-10)
    # Convert letters to numbers (A=10, B=11, ..., Z=35) and keep digits as is.
    # Then, concatenate them into a single integer string.
    try:
        numeric_str = "".join(str(int(c, 36)) for c in lei)
        # The check is valid if the number mod 97 equals 1.
        return int(numeric_str) % 97 == 1
    except (ValueError, TypeError):
        # Should not happen if regex passes, but good for safety
        return False

def contains_valid_lei(text: str) -> bool:
    """
    Searches a block of text and returns True if at least one valid LEI is found.

    Args:
        text: The string to search for LEIs.

    Returns:
        True if a valid LEI is found, otherwise False.
    """
    # Use a regex to find all 20-character alphanumeric strings
    # We convert text to uppercase to match the [A-Z0-9] character set
    potential_leis = re.findall(r'[A-Z0-9]{20}', text.upper())

    # Check if any of the found candidates is a valid LEI
    for candidate in potential_leis:
        if is_valid_lei(candidate):
            return True # Found one, no need to check further

    return False
