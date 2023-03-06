def print_progress(i, total_interation, bar_length=20):
    progress = (i + 1) / total_interation
    filled_length = int(progress * bar_length)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r|{bar}| {progress:.0%}', end='')
    if i + 1 == total_interation:
        print()
