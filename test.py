import numpy as np
import matplotlib.pyplot as plt

def sin_function(size):
    t = np.linspace(0, 20, size)
    signal = 5 * np.sin(t)
    return signal

def linear_function(size):
    t = np.linspace(0, 20, size)
    signal = 2 * t + 1
    return signal

def data_generator(size, base_function, noise_function=np.random.normal):
    X = base_function(size)
    print(X)
    y = X + noise_function(0, 1, size)
    return X, y

def SSA(X, l):
    n = len(X)
    k = n - l + 1
    hankel_matrix = np.zeros((k, l))
    for i in range(k):
        hankel_matrix[i] = X[i:i+l]
    U, S, Vt = np.linalg.svd(hankel_matrix)
    final_matrix = np.zeros((len(U), len(Vt)))
    for i in range(len(S)):
        if S[i] > 20:
            final_matrix += S[i] * np.outer(U[:, i], Vt[i])
    final_signal = np.zeros(n)
    for i in range(n):
        count = 0
        for j in range(max(0, i-l+1), min(i+1, k)):
            final_signal[i] += final_matrix[j, i-j]
            count += 1
        if count > 0:
            final_signal[i] /= count
    return final_signal

def show(x, X, y, output_signal):
    plt.figure()
    plt.plot(x, X, label="Original Signal", marker='o', color='blue')
    plt.plot(x, y, label="Input Signal", marker='s', color='orange')
    plt.plot(x, output_signal, label="SSA Output Signal", marker='x', color='red')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("SSA Decomposition")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    size = 100
    x = np.linspace(0, 5, size)
    print("Time Points:", x)
    l = 10
    X, y = data_generator(size, sin_function)
    print("Input Signal:", X)
    output_signal = SSA(y, l)
    print("Output Signal:", output_signal)
    
    show(x, X, y, output_signal)