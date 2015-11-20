"""Training Classifiers via SGD
对training数据集进行训练，将训练得到的向量应用到test集进行分类，并评价分类结果
SGD采用hinge loss和log loss两种方法，
测试集有2个，其中每个都含有traning集和test集，
最后对每个test集，每种loss方法，
屏幕输出10个采样点得到的error rate，
并生成4张折线图反映error rate的变化情况
"""
"""
需要numpy库提供矩阵运算支持，
需要matplotlib库提供生成图表功能。
代码在
Mac OS X EL CAPITAN (10.11.1)
Python 3.5(x64)
Numpy 1.10.1
Matplotlib 1.5.0
下测试通过
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt


class Training():

    """Training Classifiers via SGD
    """

    def __init__(self, training_file, test_file):
        """初始化
        调用read_from_file()读取training集和test集并处理
        """
        self.train_data, self.train_label = self.read_from_file(training_file)
        self.test_data, self.test_label = self.read_from_file(test_file)

    def get_example_num(self):
        """得到training集的example数
        即行数
        """
        return len(self.train_data)

    def read_from_file(self, filename):
        """读取并处理文件
        将数据集文件读取为data和label两部分
        data存储为二维矩阵
        label为每行最后一个，即其真实类别
        """
        data = []
        label = []
        with open(filename, 'r') as f:
            for line in f:
                t = [int(x) for x in line.split(',')[:-1]]  # 最后一个值是label
                l = int(line.rsplit(',', 1)[-1].strip())
                data.append(t)
                label.append(l)
        data = np.array(data)  # 转换为numpy.array类型
        label = np.array(label)
        return data, label

    def pegasos_hinge_loss(self, data, label, lam, ts, T):
        """hinge loss
        data: 数据矩阵
        label: label list
        lam: pegasos参数
        ts: 采样点list
        T: pegasos参数，最大循环次数
        对training集进行训练，得到w向量
        同时根据ts中值进行采样
        """
        max_row = len(data)-1
        w = np.zeros(len(data[0]))  # w向量初始值为全0
        index_t = 0
        ws = []  # 存储每个采样点得到的w向量
        for t in range(1, T+1):
            rand = random.randint(0, max_row)  # 均匀随机选择training向量
            x = data[rand]  # 以下为采用hinge loss进行训练
            y = label[rand]
            temp = y*np.inner(w, x)
            w = (1-1/t)*w
            if temp < 1:
                w = w+(1/(lam*t))*y*x
            if t == ts[index_t]:  # 到达采样点则对w进行采样
                ws.append(w)
                index_t += 1
        return ws

    def pegasos_log_loss(self, data, label, lam, ts, T):
        """log loss
        data: 数据矩阵
        label: label list
        lam: pegasos参数
        ts: 采样点list
        T: pegasos参数，最大循环次数
        对training集进行训练，得到w向量
        同时根据ts中值进行采样
        """
        max_row = len(data)-1
        w = np.zeros(len(data[0]))  # w向量初始值为全0
        index_t = 0
        ws = []  # 存储每个采样点得到的w向量
        for t in range(1, T+1):
            rand = random.randint(0, max_row)  # 均匀随机选择training向量
            x = data[rand]  # 以下为采用log loss进行训练
            y = label[rand]
            w = (1-1/t)*w+(1/(lam*t))*(y/(1+math.e**(y*np.inner(w, x))))*x
            if t == ts[index_t]:  # 到达采样点则对w进行采样
                ws.append(w)
                index_t += 1
        return ws

    def test_error(self, w, test, label):
        """根据得到的w对test集测试，得到error rate
        w: 训练得到的向量
        test: test数据集
        label: test数据集对应的label list，作为ground truth
        """
        res = np.inner(w, test)  # numpy运算，得到test集中每一行对w做内积得到的值
        wrong = 0  # 错误数
        for (t, l) in zip(res, label):  # res list和label list同时遍历
            if (t > 0 and l == -1) or (t < 0 and l == 1):  # 如果分类错误
                wrong += 1
        return wrong/len(res)  # 返回error rate

    def test_error_s(self, ws):
        """根据多个w向量，返回对应的error rate
        实际即对每个w向量调用test_error()，
        error rate作为list返回
        """
        result = []  # 存储多个error rate
        for w in ws:  # 对每个w向量调用test_error并收集返回的error rate
            test_err = self.test_error(w, self.test_data, self.test_label)
            result.append(test_err)
        return result

    def to_img(self, ys, title):
        """得到error rate折线图
        ys: 此时即error rate list
        title: 折线图标题
        此图表很多参数写死在函数里，不能够自适应
        调用后会自动在当前目录下生成title.png的图片
        """
        xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # x轴点
        plt.plot(xs, ys)  # 根据xs和ys作图
        plt.title(title)  # 添加标题
        plt.ylabel('test error')  # y轴说明
        plt.xlabel('number of iterations')  # x轴说明
        plt.axis([0.1, 1, 0, 1])  # x轴和y轴上下界
        plt.xticks(xs, [r'$0.1T$', r'$0.2T$', r'$0.3T$', r'$0.4T$', r'$0.5T$',
                        r'$0.6T$', r'$0.7T$', r'$0.8T$', r'$0.9T$', r'$1T$'])  # x轴每一点添加说明
        plt.savefig(title+".png")  # 保存到当前目录下
        plt.close()  # 清理

    def get_ts(self, T):
        """返回采样点的list
        T: 训练最大次数，集循环次数
        返回结果为
        0.1T,0.2T,0.3T,0.4T,0.5T,0.6T,0.7T,0.8T,0.9T,T
        """
        result = []  # 采样点list
        for i in range(1, 11, 1):  # 一共10个均匀采样点
            result.append(int(T*i*0.1))
        return result

    def start(self, lam, title):
        """自动执行
        lam: pegasos参数
        title: 生成图表标题
        自动进行训练并分别得到hinge loss和log loss的test error
        """
        T = 5*len(self.train_data)  # 题目要求，将T设为training集sample数的5倍
        ts = self.get_ts(T)
        test_error = self.test_error_s(
            self.pegasos_hinge_loss(self.train_data, self.train_label, lam, ts, T))
        name = title+" hinge loss"
        self.to_img(test_error, name)
        print(name+"\t", test_error)
        test_error = self.test_error_s(
            self.pegasos_log_loss(self.train_data, self.train_label, lam, ts, T))
        name = title+" log loss"
        self.to_img(test_error, name)
        print(name+"\t", test_error)
if __name__ == '__main__':
    t = Training(
        'dataset/dataset1-a8a-training.txt', 'dataset/dataset1-a8a-testing.txt')
    t.start(1e-4, "dataset 1")  # 第一个数据集
    t = Training(
        'dataset/dataset1-a9a-training.txt', 'dataset/dataset1-a9a-testing.txt')
    t.start(5e-5, "dataset 2")  # 第二个数据集
