import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')
sns.set_style('whitegrid')


def newtons_method(f, fprime, guess, epsilon=1e-10, max_iter=50):
    approx = np.zeros(max_iter + 1)
    approx[0] = guess
    iterations = 0
    while iterations < max_iter:
        approx[iterations + 1] = approx[iterations] - f(approx[iterations]) / fprime(approx[iterations])
        if np.abs(approx[iterations + 1] - approx[iterations]) < epsilon:
            break
        iterations += 1

    return approx[:iterations+1], iterations

def f(x):
    return 54 * x**6 + 45 * x**5 - 102 * x**4 - 69 * x**3 + 35 * x**2 + 16 * x - 4

def fp(x):
    return 324 * x**5 + 225 * x**4 - 408 * x**3 - 207 * x**2 + 70 * x + 16

def g(x):
    const = 1 if (x-3/4) > 0 else -1
    return const * np.power(abs(1 - 3/(4*x)),1/3)

def gp(x):
    const = 1 if (x-3/4) > 0 else -1
    return const * np.power(abs(1-3/(4*x)),-2/3) * 1/(4*x**2)

x0 = 0.7500000001
approx, iterations = newtons_method(g, gp, x0)

# zero = lambda x: 0 * x
# x = np.linspace(-1.4,1.3,100)

# plt.plot(x, f(x))
# plt.plot(x, zero(x), 'k--')
# plt.xlim([-1.4,1.3])
# 
# plt.title('Plot of $f(x) = 54x^6 + 45 x^5 - 102x^4 - 69x^3 + 35x^2 + 16x - 4$')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# 
# plt.savefig('plot0.png', figsize=(4,4), transparent=True, bbox_inches='tight')

x = [-1.5, -0.7, 0.2, 0.5, 1.2]
results = [newtons_method(f, fp, x0) for x0 in x]

for i, x0 in enumerate(x):
    print('{:0.1f} & {:0.6f} & {:d} & {:0.6f} \\\\'.format(x0,
        results[i][0][-1],
        results[i][1],
        fp(results[i][0][-1])))
    





# plt.figure(figsize=(8,6))

# plt.scatter(np.arange(0,51), np.abs(approx))
# 
# plt.semilogy()
# plt.xlim([0,30])
# 
# plt.title('Absolute iterates of Newton\'s method for $f(x) = (1 - 3/(4x))^{1/3}$.')
# plt.ylabel('Absolute iterate approximation')
# plt.xlabel('Iteration')
# 
# plt.savefig('plot.png', figsize=(4,4), transparent=True, bbox_inches='tight')

