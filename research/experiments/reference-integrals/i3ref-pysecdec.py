import os
import math
import subprocess
from pySecDec.code_writer import make_package
from pySecDec.integral_interface import IntegralLibrary
import re
from datetime import datetime

LA = 1.0
M1 = 0.3 / LA
M2 = 0.3 / LA
M3 = 0.3 / LA
P1P1 = -(0.14 / LA) ** 2
P2P2 = -(0.14 / LA) ** 2
PP = -(0.70 / LA) ** 2
M1sq, M2sq, M3sq = M1 ** 2, M2 ** 2, M3 ** 2
R2MID = PP - P1P1 - P2P2

T_MAX = 80.0
L_LOG = math.log(1.0 + T_MAX)  # численное значение — только в Python, не в FORM

A_VALS = [0, 0, 1, 1, 0, 0, 1, 1]
B_VALS = [0, 1, 0, 1, 0, 1, 0, 1]
M_VALS = [1, 1, 1, 1, 2, 2, 2, 2]
N_VALS = [2, 2, 2, 2, 3, 3, 3, 3]


def build_integrand_string(a, b, m, n):
    a1 = "x1"
    a2 = "(x2*(1-x1))"
    a3 = "((1-x1)*(1-x2))"

    D = (
        f"({a1}*{a2}*PP"
        f" + P1P1*{a2}*{a3}"
        f" + P2P2*{a1}*{a3}"
        f" + {a1}*M1sq"
        f" + {a2}*M2sq"
        f" + {a3}*M3sq)"
    )

    R2 = (
        f"(({a1})^2*P2P2"
        f" + ({a2})^2*P1P1"
        f" - {a1}*{a2}*R2MID)"
    )

    z0 = (
        f"((exp(LL*x3)-1)*{D}"
        f" + (1-exp(-LL*x3))*{R2})"
    )

    t_weight = f"(exp(LL*x3)-1)^{m} * exp(-{n}*LL*x3)"
    jac = f"(LL * exp(LL*x3) * (1-x1))"

    expr = (
        f"(({a1})^{a} * ({a2})^{b}"
        f" * {t_weight}"
        f" * exp(-({z0}))"
        f" * {jac})"
    )
    return expr


def parse_secdec_result(result_str: str) -> tuple[float, float]:
    match = re.search(r'\(\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)\s*\+/-\s*(\d+\.?\d*(?:e[+-]?\d+)?)\s*\)', result_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"Cannot parse result string: {repr(result_str)}")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_filename = f"i3ref_{timestamp}.csv"

    with open(output_filename, "w") as f_out:
        for i in range(len(A_VALS)):
            a, b, m, n = A_VALS[i], B_VALS[i], M_VALS[i], N_VALS[i]
            name = f"integral_{i}"

            integrand_expr = build_integrand_string(a, b, m, n)

            make_package(
                name=name,
                integration_variables=['x1', 'x2', 'x3'],
                real_parameters=['PP', 'P1P1', 'P2P2', 'M1sq', 'M2sq', 'M3sq', 'R2MID', 'LL'],
                regulators=['eps'],
                requested_orders=[0],
                polynomials_to_decompose=['1'],
                remainder_expression=integrand_expr,
            )

            print(f"Building C++ library for {name}...")
            custom_cxx = "c++ -Wno-missing-template-arg-list-after-template-kw -Wno-vla-cxx-extension -Wno-braced-scalar-init"
            subprocess.run(["make", "-C", name, f"CXX={custom_cxx}"], check=True)

            lib_path = os.path.join(name, f"{name}_pylink.so")
            integral = IntegralLibrary(lib_path)
            #integral.use_Vegas(flags=2, maxeval=2_000_000, epsrel=1e-4, epsabs=1e-10)
            # integral.use_Cuhre(flags=2, maxeval=10_000_000, epsrel=1e-4, epsabs=1e-10, key=13)
            integral.use_Qmc(
                transform="korobov3",
                maxeval=100_000_000,
                epsrel=1e-5,
                epsabs=1e-12,
            )

            real_params = [PP, P1P1, P2P2, M1sq, M2sq, M3sq, R2MID, L_LOG]
            str_result, str_unc, result_obj = integral(real_parameters=real_params)

            val, err = parse_secdec_result(result_obj)

            print(f"Result for index {i} (a={a}, b={b}, m={m}, n={n}):")
            print(f"  value = {val:.8e}  +/-  {err:.3e}")
            print("-" * 50)

            f_out.write(f"{val:.18e}, {err:.18e}\n")

            f_out.flush()


if __name__ == "__main__":
    main()
