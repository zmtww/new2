# new2
# 幸福指数跟GDP等很多因素有关
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

data = pd.read_csv('../data/word-happiness-report-2017.csv')
# 得到训练和测试数据
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
input_param_name = 'x'
output_param_name = 'y'
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values
x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values
plt.scatter(x_train,y_train,label='Train Data')
plt.scatter(x_test,y_test,label='Test Data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title(''happy)
plt.legend() # legend就是图例的意思
plt.show()
num_iterations=500
learning_rate=0.01
linear_regression=LinearRegression(x_train,y_train) # 实例化
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)
print('开始时的损失：',cost_history[0])
print('训练后的损失：',cost_history[-1])
plt.plot(range(num_iterations),cost_history)
plt.xlabel()
plt.ylabel()
plt.title()
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
# linspace()通过指定开始值、终值和元素个数创建表示等差数列的一维数组
y_predictions = linear_regression.predict(x_predictions)
# 画一下，代码略


























