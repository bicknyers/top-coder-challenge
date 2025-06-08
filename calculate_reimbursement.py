import sys
from generated import calc


if __name__ == "__main__":
    if len(sys.argv) != 4:
        exit(1)

    try:
        miles = float(sys.argv[2])
        rec = float(sys.argv[3])
        dur = int(sys.argv[1])
    except Exception as e:
        exit(1)

    print(calc(miles, rec, dur))
