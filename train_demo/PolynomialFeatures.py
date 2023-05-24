import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 输入数据，特征X和目标变量y
X = np.array([[1], [2], [3], [4]])
y = np.array([3, 5, 7, 9])

# 创建多项式特征
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X_poly, y)

# 进行预测
x_test = np.array([[5]])
x_test_poly = poly_features.transform(x_test)
y_pred = model.predict(x_test_poly)

# 打印预测结果
print("预测结果:", y_pred)
