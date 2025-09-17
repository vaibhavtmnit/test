import os
from typing import List

def find_line_number_by_content(
    file_path: str, 
    code_line_content: str,
    find_all: bool = False
) -> int | List[int] | None:
    """
    Finds the line number(s) of a specific line of code within a file.

    This function normalizes whitespace, so it is not sensitive to leading or
    trailing spaces in either the source file or the input string.

    Args:
        file_path: The absolute or relative path to the source file.
        code_line_content: The exact line of code to search for.
        find_all: If True, returns a list of all matching line numbers.
                  If False (default), returns the line number of the first match.

    Returns:
        - If find_all is False: An integer representing the 1-indexed line number
          of the first match, or None if not found.
        - If find_all is True: A list of all matching line numbers, or an empty
          list if no matches are found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Normalize the input code line by stripping leading/trailing whitespace.
    normalized_search_line = code_line_content.strip()
    
    found_lines = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Enumerate starts from 0, so we add 1 for human-readable line numbers.
            for line_num, line in enumerate(f, 1):
                # Normalize the current line from the file for comparison.
                normalized_file_line = line.strip()

                # Check if the normalized lines are identical.
                if normalized_file_line == normalized_search_line:
                    if not find_all:
                        # If we only need the first match, return immediately.
                        return line_num
                    else:
                        # Otherwise, add it to our list of matches.
                        found_lines.append(line_num)
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return [] if find_all else None

    return found_lines if find_all else None

if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy Java file for demonstration.
    java_code_content = """
package com.example;

public class Demo {
    public static void main(String[] args) {
        System.out.println("Starting process..."); // Line 6

        int x = 100;
        calculate(x);

        System.out.println("Starting process..."); // Line 10
    }

    public static void calculate(int value) {
        System.out.println(value);
    }
}
"""
    dummy_file_path = "Demo.java"
    with open(dummy_file_path, "w") as f:
        f.write(java_code_content)

    print(f"Created dummy file: '{dummy_file_path}'\n")

    # --- Test Cases ---

    # Test 1: Find the first occurrence of a unique line.
    print("--- Test 1: Find a unique line ---")
    line_to_find_1 = "int x = 100;"
    line_number_1 = find_line_number_by_content(dummy_file_path, line_to_find_1)
    print(f"Searching for: '{line_to_find_1}'")
    print(f"Expected: 8, Found: {line_number_1}")
    print(f"Success: {line_number_1 == 8}\n")

    # Test 2: Find a line, but provide it with extra whitespace.
    print("--- Test 2: Find a line with different whitespace ---")
    line_to_find_2 = "   calculate(x);   " # Note the extra spaces
    line_number_2 = find_line_number_by_content(dummy_file_path, line_to_find_2)
    print(f"Searching for: '{line_to_find_2}'")
    print(f"Expected: 9, Found: {line_number_2}")
    print(f"Success: {line_number_2 == 9}\n")
    
    # Test 3: Find the FIRST occurrence of a duplicated line.
    print("--- Test 3: Find the first match of a duplicated line ---")
    line_to_find_3 = 'System.out.println("Starting process...");'
    line_number_3 = find_line_number_by_content(dummy_file_path, line_to_find_3)
    print(f"Searching for: '{line_to_find_3}'")
    print(f"Expected: 6, Found: {line_number_3}")
    print(f"Success: {line_number_3 == 6}\n")

    # Test 4: Find ALL occurrences of a duplicated line.
    print("--- Test 4: Find ALL matches of a duplicated line ---")
    line_to_find_4 = 'System.out.println("Starting process...");'
    line_numbers_4 = find_line_number_by_content(dummy_file_path, line_to_find_4, find_all=True)
    print(f"Searching for all instances of: '{line_to_find_4}'")
    print(f"Expected: [6, 10], Found: {line_numbers_4}")
    print(f"Success: {line_numbers_4 == [6, 10]}\n")

    # Test 5: Search for a line that does not exist.
    print("--- Test 5: Search for a non-existent line ---")
    line_to_find_5 = "int y = 200;"
    line_number_5 = find_line_number_by_content(dummy_file_path, line_to_find_5)
    print(f"Searching for: '{line_to_find_5}'")
    print(f"Expected: None, Found: {line_number_5}")
    print(f"Success: {line_number_5 is None}\n")

    # Clean up the dummy file
    os.remove(dummy_file_path)
    print(f"Removed dummy file: '{dummy_file_path}'")

