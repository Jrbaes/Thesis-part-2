import sys
nb_path = sys.argv[1]
start = int(sys.argv[2]) - 1
end = int(sys.argv[3])
lines = open(nb_path).readlines()
for i, l in enumerate(lines[start:end], start + 1):
    print(f"{i}: {repr(l)}")
