accented_to_unaccented = {
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
}


def replace_accented_with_quote(input_str):
    result = []
    for char in input_str:
        if char in accented_to_unaccented:
            result.append(accented_to_unaccented[char] + "'")
        else:
            result.append(char)

    return "".join(result)
