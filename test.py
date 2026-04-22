import numpy as np

def sin_function(size):
    t = np.linspace(0, 2, size)
    signal = 5 * np.sin(2 * np.pi * t)
    return signal

def linear_function(size):
    t = np.linspace(0, 10, size)
    signal = 2 * t + 1
    return signal

def data_generator(size, base_function, noise_function=np.random.normal):
    X = base_function(size)
    print(X)
    y = X
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
        if S[i] > 1:
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

print(1)
if __name__ == "__main__":
    size = 21
    l = 10
    X, y = data_generator(size, linear_function)
    print("Input Signal:", X)
    output_signal = SSA(X, l)
    print("Output Signal:", output_signal)
    