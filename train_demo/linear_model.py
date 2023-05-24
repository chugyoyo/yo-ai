import numpy as np
from sklearn.linear_model import LinearRegression

# 输入数据，特征X和目标变量y
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 5, 6, 7])

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 进行预测
x_test = np.array([[5]])
y_pred = model.predict(x_test)

# 打印预测结果
print("预测结果:", y_pred)
