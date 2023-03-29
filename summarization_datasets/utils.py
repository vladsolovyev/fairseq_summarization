def write_to_file(lines, file):
    with open(file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write("{}\n".format(line))
