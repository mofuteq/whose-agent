Rust is a great choice, but for this small CLI Python will be simpler and easier to maintain. Here is a standard-library-only Python version:

```python
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: linecount.py <path>", file=sys.stderr)
        return 2

    with open(sys.argv[1], "r", encoding="utf-8") as file:
        print(sum(1 for _ in file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
