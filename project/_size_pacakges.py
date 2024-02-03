# Run `python pipsize.py` in Terminal to show size of pip packages
# Credits: https://stackoverflow.com/a/67914559/11067496

import os
import pkg_resources

sort_in_descending = True   # Show packages in descending order


def calc_container(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


total_size = 0
max_size = 0
max_dist_length = 0
dists = [d for d in pkg_resources.working_set]
dists_with_size = {}

for dist in dists:
    try:
        max_dist_length = max(max_dist_length, len(str(dist)))
        path = os.path.join(dist.location, dist.project_name)
        size = calc_container(path)
        total_size += size
        max_size = max(max_size, size)
        dists_with_size[size] = dist
    except OSError:
        '{} no longer exists'.format(dist.project_name)

# Sort packages size
dists_with_size = dict(sorted(dists_with_size.items(), reverse=sort_in_descending))


def str_spacer(name: str, max_len: int = max_dist_length) -> str:
    n_spaces = max_len - len(str(name))
    return f"{n_spaces * ' '}"


def human_readable_size(size: int, decimal_places: int = 2, max_unit: str = "PiB"):
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']

    if max_unit not in units:
        raise ValueError(f"specified max unit not in available units. Available units: {units}")

    for unit in units:
        if size < 1024.0 or unit == max_unit:
            break
        size /= 1024.0

    return f"{size:.{decimal_places}f} {unit}"


def table_printer(text: str, size: int):
    print(f"{text} {str_spacer(text)}{human_readable_size(size, max_unit='MiB')}")


# print total statement
table_printer("TOTAL", total_size)
max_size_text = human_readable_size(max_size, max_unit="MiB")
print("=" * (1 + max_dist_length + len(max_size_text)))

# print size for each distro
count_small_libs = 0
small_lib_size = 0
for size, dist in dists_with_size.items():
    if size/1000000 > 1.0:
        table_printer(dist, size)
    else:
        count_small_libs += 1
        small_lib_size += size

# print remaining size for small distros
small_lib_text = f"{count_small_libs} libs smaller than 1.0 MB"
print()
table_printer(small_lib_text, small_lib_size)