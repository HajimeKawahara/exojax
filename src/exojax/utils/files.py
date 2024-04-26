import glob
from pathlib import Path

def find_files_by_extension(directory_path, extension):
    """
    Finds all files with the given extension in the specified directory.

    Args:
        directory_path (str): Path to the directory where to search for files.
        extension (str): File extension to search for. Include the dot, e.g., '.txt'.

    Returns:
        list: A list of paths to the files found with the specified extension.

    Examples:
        >>> directory_path = '/path/to/your/directory'
        >>> extension = '.txt'
        >>> files = find_files_by_extension(directory_path, extension)
    """
    pattern = f"{directory_path}/*{extension}"
    files = glob.glob(pattern)
    return files


def get_file_names_without_extension(file_paths):
    """
    Extracts the file names without extensions from a list of file paths.

    Args:
        file_paths (list): A list of file paths as strings.

    Returns:
        list: A list of file names without extensions.

    Examples:
        >>> file_paths = ["/home/kawahara/A.txt", "/home/kawahara/B.txt"]
        >>> file_names = get_file_names_without_extension(file_paths)
        >>> print(file_names) # ["A", "B"]
    """
    return [Path(path).stem for path in file_paths]

