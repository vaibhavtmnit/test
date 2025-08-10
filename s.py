def has_substring_in_list(string_list: list[str]) -> bool:
    """
    Checks if any string in a list is a substring of any other string in the list.

    Args:
        string_list: A list of strings.

    Returns:
        True if a substring relationship exists, False otherwise.
    """
    # Get the number of strings in the list
    n = len(string_list)

    # We need at least two strings to compare
    if n < 2:
        return False

    # Iterate through each string in the list using its index 'i'
    for i in range(n):
        # Iterate through each string again using its index 'j' to compare with the string at 'i'
        for j in range(n):
            # Skip comparing a string with itself
            if i == j:
                continue

            # Check if the string at index 'i' is a substring of the string at index 'j'
            # The 'in' operator in Python efficiently handles this check.
            if string_list[i] in string_list[j]:
                # If a substring is found, we can immediately return True
                print(f"Found: '{string_list[i]}' is a substring of '{string_list[j]}'")
                return True

    # If the loops complete without finding any substrings, return False
    return False

# --- Example Usage ---

# Example 1: Should return True because "app" is in "apple"
list1 = ["apple", "banana", "app", "cherry"]
print(f"Checking list: {list1}")
result1 = has_substring_in_list(list1)
print(f"Result: {result1}\n") # Output: True

# Example 2: Should return True because "cat" is in "concatenate"
list2 = ["dog", "concatenate", "bird", "cat"]
print(f"Checking list: {list2}")
result2 = has_substring_in_list(list2)
print(f"Result: {result2}\n") # Output: True

# Example 3: Should return False because no string is a substring of another
list3 = ["hello", "world", "python", "code"]
print(f"Checking list: {list3}")
result3 = has_substring_in_list(list3)
print(f"Result: {result3}\n") # Output: False

# Example 4: An empty list
list4 = []
print(f"Checking list: {list4}")
result4 = has_substring_in_list(list4)
print(f"Result: {result4}\n") # Output: False

# Example 5: A list with one item
list5 = ["single"]
print(f"Checking list: {list5}")
result5 = has_substring_in_list(list5)
print(f"Result: {result5}\n") # Output: False




def reconstruct_from_string(concatenated_string: str, valid_substrings: list[str]) -> list[str]:
    """
    Breaks a concatenated string into a list of valid substrings.

    This function works by iteratively finding which of the valid substrings
    matches the beginning of the remaining concatenated string. It prioritizes
    longer substrings to ensure correct parsing (e.g., "apple,pie" is matched
    before "apple").

    Args:
        concatenated_string: The single string made of other strings joined together.
        valid_substrings: A list of the original strings that are valid components.

    Returns:
        A list of the reconstructed substrings in the order they appeared.
        Returns an empty list if the string cannot be fully parsed.
    """
    # Sort substrings by length in descending order. This is crucial to ensure
    # we match the longest possible substring first (greedy approach).
    # For example, if we have "apple" and "applepie", we want to check for
    # "applepie" first.
    sorted_substrings = sorted(valid_substrings, key=len, reverse=True)

    reconstructed_list = []
    current_index = 0
    n = len(concatenated_string)

    # Continue as long as we haven't parsed the whole string
    while current_index < n:
        match_found = False
        # Check each valid substring to see if it matches the start of our remaining string
        for sub in sorted_substrings:
            if concatenated_string.startswith(sub, current_index):
                # If a match is found, add it to our list
                reconstructed_list.append(sub)
                # Move the index forward by the length of the matched substring
                current_index += len(sub)
                match_found = True
                # Break the inner loop and start the next search from the new index
                break
        
        # If we went through all valid substrings and found no match,
        # the string cannot be fully reconstructed.
        if not match_found:
            print(f"Error: Cannot parse the rest of the string starting at index {current_index}.")
            print(f"Unparsed part: '{concatenated_string[current_index:]}'")
            return []

    return reconstructed_list

# --- Example Usage ---

# Scenario 1: Substrings with commas in them
original_list1 = ["hello, world", "python", "is fun"]
s1 = "hello, worldpythonis fun"
print(f"Original list: {original_list1}")
print(f"Concatenated string: '{s1}'")
result1 = reconstruct_from_string(s1, original_list1)
print(f"Reconstructed list: {result1}\n")
# Expected: ['hello, world', 'python', 'is fun']

# Scenario 2: Ambiguity handled by longest-first match
original_list2 = ["app", "apple", "pie"]
s2 = "applepieapp"
print(f"Original list: {original_list2}")
print(f"Concatenated string: '{s2}'")
result2 = reconstruct_from_string(s2, original_list2)
print(f"Reconstructed list: {result2}\n")
# Expected: ['apple', 'pie', 'app']

# Scenario 3: Parsing fails because a part is not in the original list
original_list3 = ["go", "lang", "rust"]
s3 = "golangjava"
print(f"Original list: {original_list3}")
print(f"Concatenated string: '{s3}'")
result3 = reconstruct_from_string(s3, original_list3)
print(f"Reconstructed list: {result3}\n")
# Expected: [] (because "java" is not a valid substring)
