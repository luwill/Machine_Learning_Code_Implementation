# 机器学习 公式推导与代码实现
李航老师的《统计学习方法》和周志华老师的西瓜书《机器学习》一直国内机器学习领域的经典教材。本书在这两本书理论框架的基础上，补充了必要的代码实现思路和逻辑过程。

本书在对全部机器学习算法进行分类梳理的基础之上，分别对监督学习单模型、监督学习集成模型、无监督学习模型、概率模型4个大类26个经典算法进行了相对完整的公式推导和必要的代码实现，旨在帮助机器学习入门读者完整地掌握算法细节、实现方法以及内在逻辑。本书可作为《统计学习方法》和西瓜书《机器学习》的补充材料。

---
### 使用说明
本仓库为《机器学习 公式推导与代码实现》一书配套代码库，相较于书中代码而言，仓库代码随时保持更新和迭代。目前仓库只开源了全书的代码，全书内容后续也会在仓库中开源。本仓库已经根据书中章节将代码分目录整理好，读者可直接点击相关章节使用该章节代码。

---
### 纸质版
<img 
src="https://github.com/luwill/louwill-python-learning/raw/master/cover.jpg"
width = "280" height = "350">
<br>
<div style="color: #999;
font-size:11px;
padding: 2px;"></div>

购买链接：[京东](https://item.jd.com/13581834.html) | [当当](http://product.dangdang.com/29354670.html)

---
### 配套PPT
为方便大家更好的使用本书，本书也配套了随书的PPT，购买过纸质书的读者可以在机器学习实验室公众号联系作者获取。

<img 
src="https://github.com/luwill/Machine_Learning_Code_Implementation/blob/master/pic/ppt_1.png"
width = "534" height = "300">
<br>
<div style="color: #999;
font-size:11px;
padding: 2px;">第1章示例</div>


<img 
src="https://github.com/luwill/Machine_Learning_Code_Implementation/blob/master/pic/ppt_2.png"
width = "534" height = "300">
<br>
<div style="color: #999;
font-size:11px;
padding: 2px;">第2章示例</div>


<img 
src="https://github.com/luwill/Machine_Learning_Code_Implementation/blob/master/pic/ppt_3.png"
width = "534" height = "300">
<br>
<div style="color: #999;
font-size:11px;
padding: 2px;">第7章示例</div>

<img 
src="https://github.com/luwill/Machine_Learning_Code_Implementation/blob/master/pic/ppt_4.png"
width = "534" height = "300">
<br>
<div style="color: #999;
font-size:11px;
padding: 2px;">第12章示例</div>


<img 
src="https://github.com/luwill/Machine_Learning_Code_Implementation/blob/master/pic/ppt_5.png"
width = "534" height = "300">
<br>
<div style="color: #999;
font-size:11px;
padding: 2px;">第23章示例</div>


---
### 配套视频讲解（更新中）
为了帮助广大读者更好地学习和掌握机器学习的一般理论和方法，笔者在PPT基础上同时在为全书配套讲解视频。包括模型的公式手推和代码的讲解。

第一章：[机器学习入门](https://www.bilibili.com/video/BV1jR4y1A7aH#reply112207884144)

---
### 全书勘误表
勘误表：[勘误表](https://github.com/luwill/Machine_Learning_Code_Implementation/blob/master/Errata/Errata.md)

---
### 代码修订说明（2026.04）

本仓库代码已完成两轮系统性的代码 review 和优化，主要修订如下：

**结构调整**
- 创建 `mlbook/` 共享库，消除 Ch7/Ch11/Ch12/Ch15 中重复的 `utils.py` 和 `cart.py`（6 个冗余文件统一为 2 个共享文件）
- 添加 `requirements.txt`、`.gitignore`、`pyproject.toml` 等基础设施文件
- 添加 `tests/` 测试目录，覆盖共享库和关键算法的 25 个测试用例

**Bug 修复**
- Ch5 LDA：修复 `calc_cov` 错误标准化导致类内散度矩阵计算错误（准确率 0.85→1.0）
- Ch25 MCMC：修复 Gibbs 采样逻辑错误（`p_xy` 始终传入 y=-1 而非状态转移值）
- Ch3 逻辑回归：修复 `accuracy` 函数 O(n²) 循环、添加交叉熵 log clipping 防 NaN
- Ch23 HMM：修复前向算法和维特比算法中硬编码状态数 N=4 的问题
- perceptron.py：补充缺失的 `initialize_with_zeros` 方法

**API 现代化**
- 修复 `sklearn.datasets.samples_generator` → `sklearn.datasets.make_blobs`
- 修复 `np.float` → `np.float64`、`np.matrix` → 标准数组
- 修复 `normed=1` → `density=True`（matplotlib）
- Ch1 LogisticRegression 添加 `max_iter=200` 避免收敛警告

**代码质量**
- 共享库 `mlbook/decision_tree/` 添加完整类型注解
- 修复多处拼写错误（`missclassification`、`initilize_with_zeros`）
- Ch19 SVD：修复 Windows 硬编码路径为 `os.path.join`

详见 commit 历史及 `tests/` 目录。

---
### LICENSE
本项目采用[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)进行许可。
