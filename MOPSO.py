# 多目标粒子群优化算法 (MOPSO)

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

# 配置中文字体支持
try:
    # 获取系统字体路径
    import matplotlib.font_manager as fm
    
    # 查找中文字体
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name.lower()
        if 'simhei' in font_name or 'microsoft yahei' in font_name or 'simsun' in font_name:
            chinese_fonts.append(font.fname)
    
    if chinese_fonts:
        # 使用找到的第一个中文字体
        font_prop = fm.FontProperties(fname=chinese_fonts[0])
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"中文字体配置成功: 使用字体 {font_prop.get_name()}")
    else:
        # 回退到默认配置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("中文字体配置: 使用默认字体配置")
except Exception as e:
    print(f"中文字体配置警告: {e}")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用回退字体配置")

# ==================== 辅助函数定义 ====================
#初始化位置或速度
def ini(size, lower_bound, upper_bound, dim):
    """初始化位置或速度"""
    # 支持标量和数组形式的边界
    if np.isscalar(lower_bound):
        lower_bound = np.full(dim, lower_bound)
    if np.isscalar(upper_bound):
        upper_bound = np.full(dim, upper_bound)
    
    return np.random.uniform(lower_bound, upper_bound, (size, dim))

#根据位置计算适应度
def fun(position):
    """目标函数 - 改进的测试函数，产生真实的帕累托前沿"""
    # 这是一个改进的测试函数，产生两个相互冲突的目标
    # 假设位置向量有2个维度（x1, x2）
    # 第一个目标：最小化 x1
    # 第二个目标：最小化 (1 + x2) * (1 - (x1/(1+x2))**2)
    # 这样会产生一个凸的帕累托前沿
    
    # 确保位置向量至少有2个维度
    if len(position) < 2:
        # 如果维度不足，扩展位置向量
        pos = np.zeros(2)
        pos[:len(position)] = position
    else:
        pos = position[:2]  # 只使用前两个维度
    
    x1 = pos[0]
    x2 = pos[1]
    
    # 将变量限制在合理范围内
    x1 = np.clip(x1, 0, 1)
    x2 = np.clip(x2, 0, 5)
    
    # 第一个目标：最小化 x1
    f1 = x1
    
    # 第二个目标：最小化 (1 + 9*x2) * (1 - sqrt(x1/(1+9*x2)))
    # 这是一个经典的ZDT1测试函数形式，会产生凸的帕累托前沿
    g = 1 + 9 * x2
    h = 1 - np.sqrt(x1 / g)  # 使用平方根，这是ZDT1的标准形式
    f2 = g * h
    
    return np.array([f1, f2])

#判断fitness1是否被fitness2支配
def is_dominated(fitness1, fitness2):
    """判断fitness1是否被fitness2支配"""
    # 如果fitness1在所有目标上都不优于fitness2，并且在至少一个目标上劣于fitness2，则fitness1被fitness2支配
    return all(fitness1 <= fitness2) and any(fitness1 < fitness2)

# 计算适应度向量之间的拥挤距离
def calculate_crowding_distance(fitness_vectors):
    """计算适应度向量之间的拥挤距离"""
    # 拥挤距离是一个衡量解在目标空间中密度的指标，距离越大表示解所在区域越稀疏
    num_solutions, num_objectives = fitness_vectors.shape  # 获取解的数量和目标函数的数量
    distance = [0] * num_solutions  # 初始化拥挤距离列表
    epsilon = 1e-10  # 定义一个很小的值，用于避免除以零的情况
    
    for m in range(num_objectives):  # 对于每个目标函数
        sorted_indices = np.argsort(fitness_vectors[:, m])  # 根据第m个目标函数的值对解进行排序
        distance[sorted_indices[0]] = float('inf')  # 将第一个解的拥挤距离设置为无穷大
        distance[sorted_indices[-1]] = float('inf')  # 将最后一个解的拥挤距离设置为无穷大
        
        max_value = np.max(fitness_vectors[:, m])
        min_value = np.min(fitness_vectors[:, m])
        normalization_factor = max_value - min_value
        
        if normalization_factor == 0:
            normalization_factor += epsilon  # 避免除以零的情况
            
        for i in range(1, num_solutions - 1):  # 对于排序后的每个解（除了第一个和最后一个）
            next_fitness = fitness_vectors[sorted_indices[i + 1], m]  # 获取下一个解的值
            prev_fitness = fitness_vectors[sorted_indices[i - 1], m]  # 获取上一个解的值
            # 计算并累加归一化拥挤距离
            distance[sorted_indices[i]] += (next_fitness - prev_fitness) / normalization_factor
    
    return distance  # 返回所有解的拥挤距离列表

#更新存档函数
def update_archive(archive, new_position, max_archive_size):
    """将新位置加入存档，并保持存档中只有非支配解，且不超过最大容量"""
    new_fitness = fun(new_position)  # 计算新位置的适应度
    
    # 首先检查新位置是否被archive中的任何解支配
    is_none_dominated = True  # 先假设 new_position是非支配的
    to_remove = []  # 记录被支配的解的索引
    
    for i in range(len(archive)):
        archive_fitness = archive[i][1]  # 获取archive中第i个解的适应度向量
        
        if is_dominated(archive_fitness, new_fitness):  # 如果archive中的解支配new_position
            is_none_dominated = False  # new_position不是非支配的
            break
        elif is_dominated(new_fitness, archive_fitness):  # 如果new_position支配archive中的解
            to_remove.append(i)  # 记录被支配的解的索引
    
    # 更新存档
    # 移除被new_fitness支配的解
    for i in sorted(to_remove, reverse=True):
        del archive[i]
    
    # 将new_position和它的适应度加入archive
    if is_none_dominated:
        archive.append([new_position, new_fitness])
    
    # 如果archive超过最大容量，进行拥挤距离筛选
    if len(archive) > max_archive_size:
        # 从存档中提取所有适应度向量
        fitness_vectors_list = [archive[i][1] for i in range(len(archive))]
        fitness_vectors = np.array(fitness_vectors_list)  # 转换为numpy数组
        
        # 计算适应度向量之间的拥挤距离
        distance = calculate_crowding_distance(fitness_vectors)
        
        # 根据拥挤距离对存档中的解进行排序，保留拥挤距离较大的前max_archive_size个解
        sorted_indices = np.argsort(distance)[::-1]  # 根据拥挤距离降序排序
        archive = [archive[i] for i in sorted_indices[:max_archive_size]]
    
    return archive  # 返回更新后的存档

# ==================== 主程序 ====================

# 定义边界参数 - 对于ZDT1函数，x1在[0,1]，x2在[0,5]
lb = np.array([0, 0])  # 下界 - x1: [0,1], x2: [0,5]
ub = np.array([1, 5])   # 上界 - x1: [0,1], x2: [0,5]
v_min = -0.5*np.array([1, 1])  # 最小速度
v_max = 0.5*np.array([1, 1])   # 最大速度
archive_size = 50  # 存档大小

num_objectives = 2  # 目标函数的数量
pop = 100  # 种群数量
dim = 2  # 问题的维度（ZDT1函数需要2个维度：x1和x2）

# 初始化粒子位置和速度
X = ini(pop, lb, ub, dim)  # 初始化位置
v = ini(pop, v_min, v_max, dim)  # 初始化速度

# 初始化个体历史最优位置和最优适应度值
pBest = X.copy()  # 初始时个体最优位置就是初始位置
pBest_fitness = np.zeros((pop, num_objectives))  # 初始化适应度矩阵

for i in range(pop):
    pBest_fitness[i, :] = fun(X[i, :])  # 计算初始适应度

# 初始化非支配解集
archive = []

# 主迭代循环
maxIter = 100  # 最大迭代次数

for iter in range(maxIter):
    print(f"迭代次数={iter+1}/{maxIter}")
    
    # 更新个体最优解
    for i in range(pop):
        current_f = fun(X[i, :])  # 计算当前点的函数值
        
        if is_dominated(current_f, pBest_fitness[i, :]):  # 判断当前点是否能够支配历史最优
            pBest[i, :] = X[i, :]
            pBest_fitness[i, :] = current_f
    
    # 更新外部存档
    for i in range(pop):
        archive = update_archive(archive, X[i, :], archive_size)
    
    # 更新粒子速度和位置
    for i in range(pop):
        # 从存档中随机选择一个非支配解作为全局最优解
        if len(archive) > 0:
            gBest = random.choice(archive)[0]  # archive[i][0]表示第i个解的位置
        else:
            gBest = random.choice(X)  # 如果存档为空，随机选择一个粒子的位置作为全局最优解
        
        # 更新速度和位置
        w = 0.5  # 惯性权重
        c1 = 1.0  # 个体学习因子
        c2 = 1.0  # 社会学习因子
        r1 = np.random.rand()  # 生成一个[0,1]之间的随机数
        r2 = np.random.rand()  # 生成一个[0,1]之间的随机数
        
        v[i, :] = w * v[i, :] + c1 * r1 * (pBest[i, :] - X[i, :]) + c2 * r2 * (gBest - X[i, :])  # 更新速度
        X[i, :] = X[i, :] + v[i, :]  # 更新位置
        X[i, :] = np.clip(X[i, :], lb, ub)  # 将位置限制在边界内
    
    # 动态绘制当前pareto前沿（每10次迭代更新一次）
    if (iter + 1) % 10 == 0:
        f1 = []
        f2 = []
        for i in range(len(archive)):
            f1.append(archive[i][1][0])  # 第一个目标函数值
            f2.append(archive[i][1][1])  # 第二个目标函数值
        
        print(f"迭代 {iter+1}: 存档大小={len(archive)}, 正在更新图形...")
        
        # 如果是第一次绘图，创建图形和坐标轴
        if iter == 9:
            plt.ion()  # 开启交互模式
            fig, ax = plt.subplots(figsize=(10, 8))
            scat = ax.scatter(f1, f2, alpha=0.7, label=f'迭代 {iter+1}')
            ax.set_xlabel('目标函数1 (f1) - 最小化')
            ax.set_ylabel('目标函数2 (f2) - 最小化')
            ax.set_title(f'动态Pareto前沿更新 (存档大小: {len(archive)})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置固定的坐标轴范围（基于第一次绘图的数据）
            # 添加一些边距
            x_margin = (max(f1) - min(f1)) * 0.1 if len(f1) > 1 else 0.1    # 如果f1中有多个值，计算边距；否则使用默认边距
            y_margin = (max(f2) - min(f2)) * 0.1 if len(f2) > 1 else 0.1    # 如果f2中有多个值，计算边距；否则使用默认边距
            
            x_min = min(f1) - x_margin if len(f1) > 0 else 0        # 如果f1中有值，设置x轴最小值；否则使用默认值
            x_max = max(f1) + x_margin if len(f1) > 0 else 1        # 如果f1中有值，设置x轴最大值；否则使用默认值
            y_min = min(f2) - y_margin if len(f2) > 0 else 0        # 如果f2中有值，设置y轴最小值；否则使用默认值
            y_max = max(f2) + y_margin if len(f2) > 0 else 1        #
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 禁用科学计数法
            ax.ticklabel_format(style='plain', axis='both')
            
            plt.pause(0.5)  # 增加暂停时间以便观察
            print(f"迭代 {iter+1}: 图形已创建")
        else:
            # 更新现有图形
            ax.clear()
            ax.scatter(f1, f2, alpha=0.7, label=f'迭代 {iter+1}')
            ax.set_xlabel('目标函数1 (f1) - 最小化')
            ax.set_ylabel('目标函数2 (f2) - 最小化')
            ax.set_title(f'动态Pareto前沿更新 (存档大小: {len(archive)})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 保持固定的坐标轴范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # 禁用科学计数法
            ax.ticklabel_format(style='plain', axis='both')
            
            plt.pause(0.5)  # 增加暂停时间以便观察
            print(f"迭代 {iter+1}: 图形已更新")

# 最终结果输出
print("\n" + "="*50)
print("优化完成!")
print(f"最终存档大小: {len(archive)}")
print("Pareto前沿解:")

for i, solution in enumerate(archive[:10]):  # 只显示前10个解
    position = solution[0]
    fitness = solution[1]
    print(f"解 {i+1}:")
    print(f"  位置: {position}")
    print(f"  适应度: f1={fitness[0]:.4f}, f2={fitness[1]:.4f}")

# 绘制最终的Pareto前沿
if len(archive) > 0:
    f1 = [archive[i][1][0] for i in range(len(archive))]
    f2 = [archive[i][1][1] for i in range(len(archive))]
    
    # 如果之前已经创建了图形，更新它；否则创建新图形
    if 'ax' in locals():
        ax.clear()
        ax.scatter(f1, f2, c='red', s=50, alpha=0.7, label='最终Pareto前沿')
        ax.set_xlabel('目标函数1 (f1) - 最小化')
        ax.set_ylabel('目标函数2 (f2) - 最小化')
        ax.set_title(f'最终Pareto前沿 (存档大小: {len(archive)})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 保持固定的坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # 禁用科学计数法
        ax.ticklabel_format(style='plain', axis='both')
        
        plt.pause(2)  # 显示最终结果2秒钟
        plt.ioff()  # 关闭交互模式
        print("\n动态可视化已完成，最终帕累托前沿图像已显示...")
        print("注意: 图像窗口将保持打开状态，请手动关闭窗口以继续程序。")
        plt.show(block=True)  # 使用block=True保持窗口打开
    else:
        plt.figure(figsize=(10, 8))
        plt.scatter(f1, f2, c='red', s=50, alpha=0.7, label='最终Pareto前沿')
        plt.xlabel('目标函数1 (f1) - 最小化')
        plt.ylabel('目标函数2 (f2) - 最小化')
        plt.title(f'最终Pareto前沿 (存档大小: {len(archive)})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
else:
    print("警告: 存档为空，没有找到非支配解!")