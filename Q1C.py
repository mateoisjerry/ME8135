# Matthew Lisondra
# ME8135 Q1C

# Import the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

# Mean and variance set
mu = 0
variance = 1
sigma = math.sqrt(variance)

# Varying x for K
x1 = np.linspace(mu - 3*sigma, mu + 3*sigma, 1)
x2 = np.linspace(mu - 3*sigma, mu + 3*sigma, 2)
x3 = np.linspace(mu - 3*sigma, mu + 3*sigma, 3)
x10 = np.linspace(mu - 3*sigma, mu + 3*sigma, 10)
x100 = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

# Varying y for K
y1 = np.transpose(stats.norm.pdf(x1, mu, sigma))*stats.norm.pdf(x1, mu, sigma)
y2 = np.transpose(stats.norm.pdf(x2, mu, sigma))*stats.norm.pdf(x2, mu, sigma)
y3 = np.transpose(stats.norm.pdf(x3, mu, sigma))*stats.norm.pdf(x3, mu, sigma)
y10 = np.transpose(stats.norm.pdf(x10, mu, sigma))*stats.norm.pdf(x10, mu, sigma)
y100 = np.transpose(stats.norm.pdf(x100, mu, sigma))*stats.norm.pdf(x100, mu, sigma)

# Plot all y for K
# plt.plot(x1, y1, label = "K=1")
# plt.plot(x2, y2, label = "K=2")
# plt.plot(x3, y3, label = "K=3")
# plt.plot(x10, y10, label = "K=10")
plt.plot(x100, y100, label = "K=100")
plt.xlabel('x') 
plt.ylabel('y=x^Tx')
plt.title("PDF of y=x^Tx for varying K=1,2,3,10,100")
plt.legend()
plt.grid()
plt.show()