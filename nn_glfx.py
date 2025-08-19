import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import pearsonr

# ===== 1. 模拟两个通道信号 =====
np.random.seed(42)
t = np.linspace(0, 10, 1000)  # 时间轴
channel1 = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(len(t))  # 通道1
channel2 = np.sin(2 * np.pi * 1 * t + np.pi/4) + 0.1 * np.random.randn(len(t))  # 通道2（有相位差）

# ===== 2. 算法1: Pearson相关系数 =====
pearson_corr, p_value = pearsonr(channel1, channel2)
print(f"Pearson相关系数: {pearson_corr:.4f}, p值: {p_value:.4e}")

# ===== 3. 算法2: 互相关分析 =====
cross_corr = correlate(channel2 - np.mean(channel2), channel1 - np.mean(channel1), mode='full')
lags = np.arange(-len(channel1)+1, len(channel1))
lag_at_max = lags[np.argmax(cross_corr)]
print(f"互相关最大值出现在延迟 {lag_at_max} 个采样点")

# ===== 4. 可视化 =====
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, channel1, label='channel1 ')
plt.plot(t, channel2, label='channel2 ')
plt.xlabel("time (s)")
plt.ylabel("Signal Amplitude")
plt.title("Original Signal")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lags, cross_corr)
plt.xlabel("Delay (Samples)")
plt.ylabel("Cross-Correlation Value")
plt.title("Cross-Correlation Analysis")
plt.axvline(lag_at_max, color='r', linestyle='--', label=f'Maximum Correlation Delay={lag_at_max}')
plt.legend()

plt.tight_layout()
plt.show()
