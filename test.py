import numpy as np
import matplotlib.pyplot as plt

def sin_function(size):
    t = np.linspace(0, 20, size)
    signal = 5 * np.sin(t)  # 振幅为5的正弦函数
    return signal

def linear_function(size):
    t = np.linspace(0, 20, size)
    signal = 2 * t + 1  # 线性函数
    return signal

def cosine_function(size):
    t = np.linspace(0, 20, size)
    signal = 3 * np.cos(2 * np.pi * t / 5)  # 周期为5
    return signal

def square_wave(size):
    t = np.linspace(0, 20, size)
    signal = np.sign(np.sin(2 * np.pi * t / 4))  # 周期为4的方波
    return signal

def sawtooth_wave(size):
    t = np.linspace(0, 20, size)
    signal = 2 * (t / 3 - np.floor(t / 3 + 0.5))  # 周期为3的锯齿波
    return signal

def exponential_decay(size):
    t = np.linspace(0, 20, size)
    signal = 10 * np.exp(-t / 5)  # 指数衰减
    return signal

def modulated_sin(size):
    t = np.linspace(0, 20, size)
    carrier = np.sin(2 * np.pi * t)  # 载波
    envelope = 1 + 0.5 * np.sin(2 * np.pi * t / 10)  # 包络，周期为10
    signal = envelope * carrier  # 调制信号
    return signal

def random_walk(size):
    steps = np.random.choice([-1, 1], size=size)
    signal = np.cumsum(steps)  # 随机游走
    return signal

def data_generator(size, base_function, noise_function=np.random.normal):
    X = base_function(size)
    y = X + noise_function(0, 0.5, size)
    return X, y

def SSA(X, l):  # 奇异谱分解
    n = len(X)
    k = n - l + 1
    hankel_matrix = np.zeros((k, l))
    for i in range(k):
        hankel_matrix[i] = X[i:i+l]  # 构建Hankel矩阵

    U, S, Vt = np.linalg.svd(hankel_matrix)
    final_matrix = np.zeros((len(U), len(Vt)))
    for i in range(len(S)):
        if S[i] > 10:
            final_matrix += S[i] * np.outer(U[:, i], Vt[i])  # 只保留前几个奇异值对应的分量

    final_signal = np.zeros(n)
    for i in range(n):
        count = 0
        for j in range(max(0, i-l+1), min(i+1, k)):
            final_signal[i] += final_matrix[j, i-j]
            count += 1
        if count > 0:
            final_signal[i] /= count  # 平均化处理，得到最终的SSA分解结果并还原回时序信号

    return final_signal

def show(x, X, y, output_signal):  # 绘制图像
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

def square_loss(y_true, y_pred):  # 均方误差损失函数
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    size = 100
    x = np.linspace(0, 20, size)
    l = 10
    X, y = data_generator(size, modulated_sin)
    output_signal = SSA(y, l)
    
    show(x, X, y, output_signal)
    
    loss = square_loss(X, output_signal)
    print(f"SSA Square Loss: {loss:.4f}")
    raw_data_loss = square_loss(X, y)
    print(f"Raw Data Square Loss: {raw_data_loss:.4f}")