def get_first_part_in_key(k, comma='.'):
    parts = k.split(comma)
    return parts[0], comma.join(parts[1:])
