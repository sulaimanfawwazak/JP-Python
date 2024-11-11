import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
x = np.linspace(-np.pi, np.pi, 200)
y = np.zeros(len(x))

labels = ['First Order', 'Third Order', 'Fifth Order', 'Seventh Order']

plt.figure(figsize=(10, 8))
for n, label in zip(range(4), labels):
  y = y + ((-1)**n * (x)**(2*n+1)) / np.math.factorial(2*n+1)
  plt.plot(x, y, label=label)

plt.plot(x, np.sin(x), 'k', label='Analytic')
plt.grid()
plt.title('Taylor Series Approximation of Various Orders')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()