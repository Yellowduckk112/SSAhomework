import numpy as np
import matplotlib.pyplot as plt

def sin_function(arange, size):
    t = np.linspace(*arange, size)
    signal = 5 * np.sin(t)  # 振幅为5的正弦函数
    return signal

def linear_function(arange, size):
    t = np.linspace(*arange, size)
    signal = 2 * t + 1  # 线性函数
    return signal

def cosine_function(arange, size):
    t = np.linspace(*arange, size)
    signal = 3 * np.cos(2 * np.pi * t / 5)  # 周期为5
    return signal

def square_wave(arange, size):
    t = np.linspace(*arange, size)
    signal = np.sign(np.sin(2 * np.pi * t / 4))  # 周期为4的方波
    return signal

def sawtooth_wave(arange, size):
    t = np.linspace(*arange, size)
    signal = 2 * (t / 3 - np.floor(t / 3 + 0.5))  # 周期为3的锯齿波
    return signal * 3

def exponential_decay(arange, size):
    t = np.linspace(*arange, size)
    signal = 10 * np.exp(-t / 5)  # 指数衰减
    return signal

def modulated_sin(arange, size):
    t = np.linspace(*arange, size)
    carrier = np.sin(2 * np.pi * t)  # 载波
    envelope = 1 + 0.5 * np.sin(2 * np.pi * t / 10)  # 包络，周期为10
    signal = envelope * carrier  # 调制信号
    return signal * 5

def data_generator(arange, size, base_function, noise_function=np.random.normal):
    X = base_function(arange, size)
    y = X + noise_function(0, 2, size)
    return X, y

def SSA(X, l):  # 奇异谱分解
    n = len(X)
    k = n - l + 1
    hankel_matrix = np.zeros((k, l))
    for i in range(k):
        hankel_matrix[i] = X[i:i+l]  # 构建Hankel矩阵

    U, S, Vt = np.linalg.svd(hankel_matrix)
    final_matrix = np.zeros((len(U), len(Vt)))

    print(S)

    max_singular_value = np.max(S)
    for i in range(len(S)):
        if S[i] > max_singular_value * 0.2:  # 在这里改变选取奇异值的范围
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
    print(x.shape, X.shape, y.shape, output_signal.shape)
    plt.figure()
    plt.plot(x, X, label="Original Signal", marker='o', color='blue')
    plt.plot(x, y, label="Input Signal", marker='.', color='orange')
    plt.plot(x, output_signal, label="SSA Output Signal", marker='x', color='red')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("SSA Decomposition")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def show_single_picture(x, X, y, output_signal):  # 绘制单张图像
    plt.figure()
    plt.plot(x, X, label="Original Signal", marker='o', color='blue')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    plt.figure()
    plt.plot(x, y, label="Input Signal", marker='.', color='orange')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    plt.figure()
    plt.plot(x, output_signal, label="SSA Output Signal", marker='x', color='red')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def square_loss(y_true, y_pred):  # 均方误差损失函数
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    size = 500  # 这里调整取点的多少
    arange = (0, 50)  # 这里调整取点的范围
    x = np.linspace(*arange, size)
    l = 200  # 这里调整SSA分解的窗口长度
    X, y = data_generator(arange, size, modulated_sin)  # 这里更改初始信号的类型,可以选择sin_function、linear_function、cosine_function、square_wave、sawtooth_wave、exponential_decay、modulated_sin等函数
    output_signal_1 = SSA(y, l)  # X是原始信号，y是带噪声的输入信号
    
    loss = square_loss(X, output_signal_1)
    print(f"SSA Square Loss: {loss:.4f}")
    raw_data_loss = square_loss(X, y)
    print(f"Raw Data Square Loss: {raw_data_loss:.4f}")
    
    show(x, X, y, output_signal_1)
    show_single_picture(x, X, y, output_signal_1)

#    output_signal_2 = SSA(output_signal_1, l)  # 对SSA的输出信号再次进行SSA分解，看看是否能进一步降低损失
#    loss = square_loss(X, output_signal_2)
#    print(f"SSA_2 Square Loss: {loss:.4f}")

#    show(x, X, output_signal_1, output_signal_2)
#    show_single_picture(x, X, output_signal_1, output_signal_2)  # 似乎不行QAQ