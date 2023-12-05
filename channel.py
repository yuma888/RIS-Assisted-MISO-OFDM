import numpy as np
import time


class Channel:
    """
    在此类中h_d_channel生成了一次用户位置，后面在生成反射信道时无需再重复生成，否则会造成两信道用户位置不匹配
    故在编写时应该先编写生成直接信道后生成反射信道等
    """
    def __init__(self, l0, l1, l2, K, N_t, M, N):
        self.l0 = l0                                                            # 直接信道的延迟抽头数目
        self.l1 = l1                                                            # 入射信道的
        self.l2 = l2                                                            # 反射信道的
        assert l0 == l1 + l2 - 1, print("error: l0 must equal l1+l2-1")         # 使得l1+l2-1=l0否则报错
        self.K = K                                                              # 用户数目
        self.N_t = N_t                                                          # 基站天线数目
        self.M = M                                                              # 反射元数目
        assert np.sqrt(M) % 1 == 0, print("error: M 必须为整数的平方")             # RIS为正方形
        self.N = N                                                              # OFDM子载波数目
        self.BS = np.array([0, 0, 20])                                          # BS的位置
        self.RIS = np.array([-10, 50, 20])                                      # RIS的位置
        self.radius = 1                                                         # 用户圆半径
        self.center = np.array([10, 50, 0])                                     # 用户圆中心点
        self.vector_BS = np.array([0, 1, 0])                                    # 导向矢量  即 基站 ULA天线的法向量
        self.vector_RIS = np.array([1, 0, 0])                                   # 导向矢量  即 RIS UPA天线的法向量  且 ris位于yoz平面
        self.USE = np.zeros((K, 3))

    def use_location(self):                      # 随机生成用户位置   返回一个(self.K * 3) 的矩阵
        theta = 2 * np.pi * np.random.random(self.K)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        z = self.center[2] + 0 * np.sin(theta)
        self.USE = np.column_stack((x, y, z))

    def distance(self, location1, location2):    # 计算两者间的的距离  输出: 如果两个输入都是向量 输出常数  如果两个输入有一个是矩阵，输出一个向量
        par = self.BS.shape                      # 随便的一个参数，用来判断维度
        if location1.shape == par and location2.shape == par:
            dis = np.linalg.norm((location1 - location2))
        else:
            dis = np.linalg.norm((location1 - location2), axis=1)
        return dis

    def ULA(self, postion1, postion2):           # 此函数为求基站的ULA响应, 其中只要输入两者直接的位置坐标即可， 终点 postion1, 起点postion2
        theta = np.arccos(((postion1 - postion2)@self.vector_BS)/(np.linalg.norm(postion1 - postion2)*np.linalg.norm(self.vector_BS)))
        ula_response = np.exp(1j * np.sin(theta) * np.pi * np.arange(0, self.N_t, 1))
        return (1/np.sqrt(self.N_t))*ula_response

    def UPA(self, postion1, postion2):           # 此函数为求基站的UPA响应, 其中只要输入两者直接的位置坐标即可， 终点 postion1, 起点postion2
        """
        :param postion1: 指终点
        :param postion2: 指起点
        :return: UPA 响应向量
        :theta: 指向量投影在xoy平面后与x轴的夹角
        : phi 指向量与z轴夹角
        """
        theta = np.arccos((((postion1 - postion2)*np.array([1, 1, 0]))@self.vector_RIS)/(np.linalg.norm((postion1 - postion2)*np.array([1, 1, 0]))*np.linalg.norm(self.vector_RIS)))  # 与Y轴的夹角
        phi = np.arccos(((postion1 - postion2)@np.array([0, 0, -1]))/(np.linalg.norm(postion1 - postion2)*np.linalg.norm(np.array([0, 0, -1]))))  # 与Z轴的夹角
        y_response = np.exp(1j * np.sin(theta) * np.sin(phi) * np.pi * np.arange(0, int(np.sqrt(self.M)), 1))
        z_response = np.exp(1j * np.cos(phi) * np.pi * np.arange(0, int(np.sqrt(self.M)), 1))
        upa_response = np.kron(y_response, z_response)
        return (1/np.sqrt(self.M))*upa_response

    def DFT(self, n, l):
        """
        :param n: 第n个子载波
        :param l: l个延时抽头
        :return:  返回一个dft向量，维度大小为1*l
        """
        dft = np.exp(-1j*2*np.pi*n*np.arange(0, l, 1)/self.N)
        return dft

    def h_d_channel(self):    # 直接信道
        """
        在此成员函数中生成了一次用户位置，后面在生成反射信道时无需再重复生成，否则会造成两信道用户位置不匹配
        故在编写时应该先编写生成直接信道后生成反射信道等
        :return: 返回一个CFR响应， 大小为self.N_t, self.N, self.K
        """
        pl_dB = -30           # path loss 路径损耗 dB形式 在参考距离1m处的
        pl = 10**(pl_dB/10)   # 转化为倍数形式
        ple = 3.5             # path loss exponent
        self.use_location()   # 生成一次用户位置，后面在生成反射信道时无需再重复生成，否则会造成两信道用户位置不匹配
        dis = self.distance(self.USE, self.BS)
        p_loss = np.sqrt(pl*(dis)**(-ple))
        H_d_CIR = (1/np.sqrt(2)) * (np.random.randn(self.N_t, self.l0, self.K) + 1j * np.random.randn(self.N_t, self.l0, self.K))   # 表示直接信道矩阵 大小为 N_t*l0*K
        for k in range(self.K):
            tau = np.sort(np.random.exponential(scale=1, size=self.l0))[::-1]
            tau = np.sqrt(tau / np.sum(tau))                                  # tau 为每个多径的权重，从大到小排序，归一化，是一个1*l0向量
            H_d_CIR[:, :, k] = p_loss[k]*np.multiply(tau, H_d_CIR[:, :, k])   # 此处为CIR大小self.N_t, self.l0, self.K
        H_d_CFR = np.zeros((self.N_t, self.N, self.K)) + 1j*np.zeros((self.N_t, self.N, self.K))
        # H_d_CIR_test = np.ones((self.N_t, self.l0, self.K))
        for k in range(self.K):
            for n in range(self.N):
                dft = self.DFT(n, self.l0)
                # print(dft, n)
                # dft_test = np.arange(0, self.l0, 1)
                # print(dft_test)
                H_d_CFR[:, n, k] = np.sum(np.multiply(dft, H_d_CIR[:, :, k]), axis=1)  # 作用是对CIR进行DFT变换得到CFR
        return H_d_CFR         # 返回一个CFR响应， 大小为self.N_t, self.N, self.K

    def trans(self, v, k):
        P = v[0] / sum(v[1:]) / k
        a = v[0]
        v = v * P
        v[0] = a
        return v / sum(v)

    def G_channel(self):     # 入射信道
        G = np.zeros((self.M, self.N_t, self.l1), dtype=complex)       # 表示入射信道矩阵  大小 M*N_t*l1
        pl_dB = -30                                                    # path loss 路径损耗 dB形式 在参考距离1m处的
        pl = 10 ** (pl_dB / 10)                                        # 转化为倍数形式
        ple = 2.2                                                      # path loss exponent
        dis = self.distance(self.RIS, self.BS)
        p_loss = np.sqrt(pl * (dis) ** (-ple))                         # 路径损耗
        K_dB = 2                                                       # los径与Nlos径的功率之比
        K = 10 ** (K_dB / 10)                                          # 转化为倍数形式
        tau = np.sort(np.random.exponential(scale=1, size=self.l1))[::-1]
        tau = self.trans(tau, K)                                       # 将其转化为LOS与Nlos功率之比为K
        tau = np.sqrt(tau)                                             # 转化为幅值
        for l in range(self.l1):                                       # 第一个路径为LOS，其他为Nlos
            if l == 0:
                G[:, :, l] = p_loss * tau[l] * self.UPA(self.RIS, self.BS).reshape(-1, 1) * self.ULA(self.RIS, self.BS).reshape(1, -1)
            else:
                G[:, :, l] = p_loss * tau[l] * (1/np.sqrt(2)) * (np.random.randn(self.M, self.N_t) + 1j * np.random.randn(self.M, self.N_t))
        return G

    def h_r_channel(self):   # 反射信道
        H_r = np.zeros((self.M, self.l2, self.K), dtype=complex)      # 表示反射信道 大小 M*l2*K
        pl_dB = -30  # path loss 路径损耗 dB形式 在参考距离1m处的
        pl = 10 ** (pl_dB / 10)  # 转化为倍数形式
        ple = 2.2  # path loss exponent
        dis = self.distance(self.USE, self.RIS)
        p_loss = np.sqrt(pl * (dis) ** (-ple))
        K_dB = 3  # los径与Nlos径的功率之比
        K = 10 ** (K_dB / 10)  # 转化为倍数形式
        for k in range(self.K):
            tau = np.sort(np.random.exponential(scale=1, size=self.l1))[::-1]
            tau = self.trans(tau, K)  # 将其转化为LOS与Nlos功率之比为K
            tau = np.sqrt(tau)  # 转化为幅值
            # print(tau, k)
            for l in range(self.l2):
                if l == 0:
                    H_r[:, l, k] = p_loss[k] * tau[l] * self.UPA(self.USE[k], self.RIS)
                else:
                    H_r[:, l, k] = p_loss[k] * tau[l] * (1 / np.sqrt(2)) * (np.random.randn(self.M) + 1j * np.random.randn(self.M))
        return H_r

    def GH_channel(self):  # 反射级联信道 CFR
        """
        此成员函数首先调用G_channel和h_r_channel生成入射信道和反射信道的CIR，最后通过公式计算出反射级联信道的CIR，再进行DFT得到反射级联信道的CFR
        :return: 返回反射级联信道的CFR，维度大小为(self.M, self.N_t, self.N, self.K)
        """
        G_l = np.zeros((self.M, self.N_t, self.l2-1), dtype=complex)
        G_r = np.zeros((self.M, self.N_t, self.l2-1), dtype=complex)
        G = self.G_channel()                                                   # 生成入射信道
        G = np.concatenate((G_l, G, G_r), axis=2)                              # G的补零信道，左右两边各补l2-1个0
        H_r = self.h_r_channel()                                               # 生成反射信道
        GH = np.zeros((self.M, self.N_t, self.l0, self.K), dtype=complex)                     # 反射级联信道的CIR
        GH_CFR = np.zeros((self.M, self.N_t, self.N, self.K), dtype=complex)   # 反射级联信道的CFR
        for k in range(self.K):                                                # 计算级联信道的CIR
            for l in range(self.l0):
                for m in range(self.l2):
                    GH[:, :, l, k] = GH[:, :, l, k] + np.diag(H_r[:, m, k].reshape(-1)) @ G[:, :, (l-m)]
        # GH = np.ones((self.M, self.N_t, self.l0, self.K))
        for k in range(self.K):                                                # 计算级联信道的CFR
            for n in range(self.N):
                dft = self.DFT(n, self.l1+self.l2-1)
                GH_CFR[:, :, n, k] = np.sum(np.multiply(dft, GH[:, :, :, k]), axis=2)   # DFT sum(e**(-j*2*pi*n*l)*GH[:, :, l, k])
        return GH_CFR

    def save(self, dir):   # 存储信道
        pass


a = time.time()
mayu = Channel(8, 6, 3, 5, 7, 9, 16)
np.random.seed(0)          # 随机数种子
num_matrices = 10000
train1 = []
train2 = []
test1 = []
test2 = []
train_number = int(num_matrices*0.9)
print(train_number)
for i in range(num_matrices):
    A = mayu.h_d_channel()
    G = mayu.GH_channel()
    print(f"当前进度:{i+1}/{num_matrices}")
    # print(A.shape)
    # print(G.shape)
    if i < train_number:
        train1.append(A)
        train2.append(G)
    else:
        test1.append(A)
        test2.append(G)
        # print(len(test1))
data_to_save_train1 = {f"matrix_{i}": matrix for i, matrix in enumerate(train1)}
data_to_save_train2 = {f"matrix_{i}": matrix for i, matrix in enumerate(train2)}
data_to_save_test1 = {f"matrix_{i}": matrix for i, matrix in enumerate(test1)}
data_to_save_test2 = {f"matrix_{i}": matrix for i, matrix in enumerate(test2)}
# # 保存数据到一个压缩文件
np.savez('DataTrainA.npz', **data_to_save_train1)
np.savez('DataTrainG.npz', **data_to_save_train2)
np.savez('DataTestA.npz', **data_to_save_test1)
np.savez('DataTestG.npz', **data_to_save_test2)
print(f"{train_number} matrices have been saved to DataTrainA.npz.")
print(f"{train_number} matrices have been saved to DataTrainG.npz.")
print(f"{num_matrices - train_number} matrices have been saved to DataTestA.npz.")
print(f"{num_matrices - train_number} matrices have been saved to DataTestG.npz.")
b = time.time()
print(f"运行时间:{b-a}")