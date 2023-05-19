# Matthew Lisondra
# ME8135 Q1C

# Import the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

# generate y based on mu, sigma, k, N
def yk(mu, sigma, k, N):
    y = np.zeros(N)
    
    for i in range(k):
        xi = np.random.normal(mu, sigma, N)
        xi2 = np.transpose(xi)*xi
        y += xi2
    
    return y

# mu, sigma = 0, 1 # mean and standard deviation
# x = np.random.normal(mu, sigma, 1000)

# yk1 = np.transpose(x)*x

# # kwargs = dict(histtype='stepfilled', normed=True, bins=4000)
# # plt.hist(yk1, **kwargs)
# # plt.show()


# plt.hist(yk1, bins=40)
# plt.show()

yk1 = yk(0,1,1,1000)
yk2 = yk(0,1,2,1000)
yk3 = yk(0,1,3,1000)
yk10 = yk(0,1,10,1000)
yk100 = yk(0,1,100,1000)

plt.hist(yk1, bins=40, label="k=1")
plt.xlabel('x') 
plt.ylabel('y=x^Tx')
plt.title("PDF of y=x^Tx for K=1 (with 1000 samples)")
plt.legend()
plt.grid()
plt.show()

plt.hist(yk2, bins=40, label="k=2")
plt.xlabel('x') 
plt.ylabel('y=x^Tx')
plt.title("PDF of y=x^Tx for K=2 (with 1000 samples)")
plt.legend()
plt.grid()
plt.show()

plt.hist(yk3, bins=40, label="k=3")
plt.xlabel('x') 
plt.ylabel('y=x^Tx')
plt.title("PDF of y=x^Tx for K=3 (with 1000 samples)")
plt.legend()
plt.grid()
plt.show()

plt.hist(yk10, bins=40, label="k=10")
plt.xlabel('x') 
plt.ylabel('y=x^Tx')
plt.title("PDF of y=x^Tx for K=10 (with 1000 samples)")
plt.legend()
plt.grid()
plt.show()

plt.hist(yk100, bins=40, label="k=100")
plt.xlabel('x') 
plt.ylabel('y=x^Tx')
plt.title("PDF of y=x^Tx for K=100 (with 1000 samples)")
plt.legend()
plt.grid()
plt.show()