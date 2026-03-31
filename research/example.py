import numpy as np
import numpy.typing as npt


def sum_vectors(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return a + b


def main() -> None:
    a = np.ones(10)
    b = np.ones(10)
    c = sum_vectors(a, b)

    print(a)
    print(b)
    print(c)


if __name__ == "__main__":
    main()
