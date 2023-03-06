def print_progress(i, total_interation, desc="", bar_length=20):
    """print ptqdm like rogress bar

    Args:
        i (int): step number starting from 0
        total_interation (int): total iteration number
        desc (str): description
        bar_length (int, optional): bar length. Defaults to 20.
    """
    progress = i / total_interation
    filled_length = int(progress * bar_length)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{desc}|{bar}| {progress:.0%}', end='')
    if i == total_interation:
        print()
