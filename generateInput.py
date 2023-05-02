import numpy as np
import matplotlib.pyplot as plt

def generate_sinusoids(num_sin):
    amplitudes = np.random.uniform(0.1, 1.0, size=num_sin)
    frequencies = np.random.uniform(1.0, 10.0, size=num_sin)
    phases = np.random.uniform(0.0, 2*np.pi, size=num_sin)
    x = np.arange(10000)
    sinusoids = np.sin(2 * np.pi * frequencies[:, np.newaxis] * x / 10000 + phases[:, np.newaxis])
    sinusoids *= amplitudes[:, np.newaxis]  # include amplitudes here
    return sinusoids

def plot_sinusoids(sinusoids):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sinusoids.T)
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Generated Sinusoids')
    plt.show()

def plot_summed_sinusoid(summed_sinusoid):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(summed_sinusoid)
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Summed Sinusoids')
    plt.show()

def taylor_decomposition(sinusoids):
    num_sin = sinusoids.shape[0]
    max_degree = 10  # maximum degree of Taylor decomposition

    # initialize arrays to store coefficients and residuals for each sinusoid
    coeffs = np.zeros((num_sin, max_degree + 1))
    residuals = np.zeros((num_sin, sinusoids.shape[1]))

    # perform Taylor decomposition for each sinusoid
    for i in range(num_sin):
        for j in range(max_degree + 1):
            coeffs[i,j] = np.sum(sinusoids[i,:] * np.power(np.arange(sinusoids.shape[1]), j)) / np.math.factorial(j)
            residuals[i,:] = residuals[i,:] + coeffs[i,j] * np.power(np.arange(sinusoids.shape[1]), j)
        
    return coeffs, residuals

# generate some sinusoids
sinusoids = generate_sinusoids(5)
#print(sinusoids)
plot_sinusoids(sinusoids)
sum_sinusoids = np.sum(sinusoids, axis=0)
plot_summed_sinusoid(sum_sinusoids)

# perform Taylor decomposition
coeffs, residuals = taylor_decomposition(sum_sinusoids)

# print coefficients for the first sinusoid
#print(coeffs[0,:])

# plot original sinusoid and Taylor series for the first sinusoid
t = np.arange(sum_sinusoids.shape[1])
plt.plot(t, sum_sinusoids[0,:], label='Original')
plt.plot(t, sum_sinusoids[0,:] + coeffs[0,0], label='Degree 0')
plt.plot(t, sum_sinusoids[0,:] + coeffs[0,0] + coeffs[0,1]*t, label='Degree 1')
plt.plot(t, sum_sinusoids[0,:] + coeffs[0,0] + coeffs[0,1]*t + coeffs[0,2]*t**2, label='Degree 2')
plt.legend()
plt.show()
