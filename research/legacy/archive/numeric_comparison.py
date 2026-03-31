import numpy as np
from scipy import integrate

BB = 100
k = -1

la = 1.0
m1 = 0.3 / la
m2 = 0.3 / la
m3 = 0.3 / la
p1p1 = -(0.14 / la) ** 2
p2p2 = -(0.14 / la) ** 2
PP = -(0.7 / la) ** 2


def testNum(a, b, m, n):
	def integrand(t, alp1, alp2):
	    global m1, m2, m3, PP, la, p1p1, p2p2 
	    nonlocal a, b, m, n
	    RR = (alp1**2) * p1p1 + (alp2**2) * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
	    DD = (alp1 * (p1p1 + m1**2) + alp2 * (p2p2 + m2**2) + (1.0 - alp1 - alp2) * (m3**2) - RR)
	    z0 = t * DD + t / (1.0 + t) * RR
	    Fz0 = np.exp(k * z0)
	    
	    return (alp1**a) * (alp2**b) * (t**m) / ((1.0 + t)**n) * Fz0

	def integrand_wrapper(X):
	    return integrand(t=X[:, 0], alp1=X[:, 1], alp2=X[:, 2])


	# --- Define integration bounds (t ∈ [0, 100], alp2 ∈ [0, 1 - alp1], alp1 ∈ [0, 1]) ---
	def integrand_t(t, alp1, alp2):
	    return integrand(t, alp1, alp2)


	def integrand_alp2(alp2, alp1):
	    # Inner integral over t from 0 to 100
	    val, _ = integrate.quad(integrand_t, 0, BB, args=(alp1, alp2), limit=200)
	    return val


	def integrand_alp1(alp1):
	    # Middle integral over alp2 from 0 to 1 - alp1
	    val, _ = integrate.quad(integrand_alp2, 0, 1 - alp1, args=(alp1,), limit=200)
	    return val


	# print("Computing I[{}, {}, {}, {}; exp(-z0)]...".format(a, b, m, n))
	alp_result, error = integrate.quad(integrand_alp1, 0, 1, limit=200)
	# print("Result: I =", alp_result)
	# print("Estimated absolute error:", error)


	def integrand_xyz(x, y, t):
	    # Transform variables
	    alp1 = x
	    alp2 = y * (1.0 - x)
	    
	    # Compute R^2 and D as before
	    RR = (alp1**2) * p1p1 + (alp2**2) * p2p2 - alp1 * alp2 * (PP - p1p1 - p2p2)
	    DD = (alp1 * (p1p1 + m1**2) + alp2 * (p2p2 + m2**2) + (1.0 - alp1 - alp2) * m3**2 - RR)
	    
	    z0 = t * DD + t / (1.0 + t) * RR
	    Fz0 = np.exp(k*z0)
	    
	    # Jacobian factor (1 - x) and α1^a α2^b
	    jac_factor = (1.0 - x)
	    alp1_power = alp1 ** a
	    alp2_power = alp2 ** b
	    
	    return alp1_power * alp2_power * jac_factor * (t ** m) / ((1.0 + t) ** n) * Fz0


	def integrand_t(t, x, y):
	    return integrand_xyz(x, y, t)


	def integrand_y(y, x):
	    val, _ = integrate.quad(integrand_t, 0, BB, args=(x, y), limit=200)
	    return val


	def integrand_x(x):
	    val, _ = integrate.quad(integrand_y, 0, 1, args=(x,), limit=200)
	    return val


	# print("Computing I using square-domain substitution (x,y,t ∈ [0,1]×[0,1]×[0,100])...")
	xy_result, error = integrate.quad(integrand_x, 0, 1, limit=200, epsabs=1e-8, epsrel=1e-6)
	# print("Result: I =", xy_result)
	# print("Estimated error:", error)

	print(f"{abs(xy_result - alp_result)}")


def main():
	A, B, M, N = [0, 1, 0, 1, 1, 0, 1, 2, 0], [0, 0, 1, 0, 1, 0, 2, 0, 0], [2, 3, 3, 1, 1, 2, 5, 5, 5], [2, 3, 3, 2, 2, 3, 5, 5, 4]
	for aa, bb, mm, nn in zip(A, B, M, N):
		print(f"N({aa}, {bb}, {mm}, {nn})")
		testNum(aa, bb, mm, nn)


if __name__ == '__main__':
	main()
