# generate-c-code-from-python

用 sympy 符号推导动力学/代价函数的雅可比与 Hessian，并**自动生成 C++ 代码**（Eigen 稀疏矩阵格式 + Bazel BUILD 规则）。

目标场景：你有一个车辆/机器人动力学模型，手写解析雅可比容易出错且难以维护。用本工具：**改 Python 符号方程 → 跑脚本 → 自动得到 C++ 头文件**。

## 目录结构

```
├── python/
│   ├── calc_jacobian_and_hessian.py   # 符号推导：动力学显式/隐式雅可比、代价雅可比/Hessian
│   ├── generate_function_c_code.py    # 符号矩阵 → C++ 代码（Eigen SparseMatrix + Triplet）
│   └── test_c_code_generate.py        # 完整示例：6 维自行车模型 → 生成 8 个头文件
├── WORKSPACE / MODULE.bazel           # Bazel 配置（Eigen 3.3.7）
├── eigen.BUILD
└── requirements.txt                   # Python 依赖（仅 sympy）
```

## 快速开始

```bash
# 1. 安装依赖（只需 sympy）
pip install -r requirements.txt

# 2. 跑示例：生成 6 维自行车模型的 C++ 代码
cd python
python test_c_code_generate.py

# 3. 产物：../c_generated_code/ 下生成 8 个 .h + BUILD
#    dynamic_continuous.h / dynamic_jacobian_x.h / dynamic_jacobian_u.h
#    cost.h / cost_jacobian_x.h / cost_jacobian_u.h
#    cost_hessian_xx.h / cost_hessian_uu.h
```

## 工作流（怎么定义自己的模型）

三步：**定义符号 → 求导 → 生成代码**。

### 1. 定义动力学和代价（sympy 表达式）

```python
from sympy import Matrix, symbols, cos, sin, tan

x_dim, u_dim = 6, 2
x = Matrix(symbols(f'x:{x_dim}'))   # 状态变量 x0..x5
u = Matrix(symbols(f'u:{u_dim}'))   # 控制变量 u0..u1

# 动力学 f(x,u)：自行车模型，L/k 可以是符号或数值
L, k = 3.0, 0.0003                  # 数值参数会被展开进生成代码
f = Matrix([
    x[4] * cos(x[2]),                                     # ẋ
    x[4] * sin(x[2]),                                     # ẏ
    x[4] * tan(x[3]) / (L * (1 + k * x[4]**2)),           # θ̇
    u[0],                                                 # δ̇
    x[5],                                                 # v̇
    u[1],                                                 # ȧ
])

# 代价（二次型）：sum(q_i * (x_i - goal_i)^2) + sum(r_i * u_i^2)
goal = Matrix(symbols(f'goal:{x_dim}'))
q = Matrix(symbols(f'q:{x_dim}'))
r = Matrix(symbols(f'r:{u_dim}'))
error = x - goal
cost = Matrix([[sum(error[i]*q[i]*error[i] for i in range(x_dim))
              + sum(u[i]*r[i]*u[i] for i in range(u_dim))]])
```

### 2. 求雅可比 / Hessian

```python
from calc_jacobian_and_hessian import calculate_dynamics_and_derivatives, calculate_cost_and_derivatives

# 动力学：显式雅可比 ∂f/∂x、∂f/∂u，以及隐式（ẋ - f）形式
Jx, Ju, Jimp_x, Jimp_u, Jimp_xdot = calculate_dynamics_and_derivatives(x_dim, u_dim, f)

# 代价：∂c/∂x、∂c/∂u、Hessian Hxx/Huu/Hux
cJx, cJu, cHxx, cHuu, cHux = calculate_cost_and_derivatives(x_dim, u_dim, cost)
```

### 3. 生成 C++ 代码

```python
from generate_function_c_code import generate_function_code, save_code_to_file

vars_xu = {'x': x_dim, 'u': u_dim}                       # 函数参数声明依据
code = generate_function_code(Jx, vars_xu, "dynamic_jacobian_x")
save_code_to_file(code, "dynamic_jacobian_x.h")          # 自动写入 ../c_generated_code/ + 更新 BUILD
```

## 生成代码长什么样

每个函数生成一个 Eigen **稀疏矩阵**头文件（只填非零元）：

```cpp
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

Eigen::SparseMatrix<double> dynamic_jacobian_x(
    const Eigen::Matrix<double, 6, 1> &x,
    const Eigen::Matrix<double, 2, 1> &u) {
    Eigen::SparseMatrix<double> out(6, 6);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(7);
    tripletList.emplace_back(0, 2, -1.0 * x[4] * std::sin(x[2]));
    tripletList.emplace_back(0, 4, std::cos(x[2]));
    // ...
    out.setFromTriplets(tripletList.begin(), tripletList.end());
    return out;
}
```

`save_code_to_file` 还会自动创建/追加 Bazel `BUILD`（`cc_library` + `@eigen` 依赖），生成的目录可直接被 Bazel 工程引用。

## 与 iLQR 求解器（VincentWong3/ilqr）的衔接

### 现状差异：Dense vs Sparse

| | ilqr 求解器（cpp/model/new_bicycle_node.h） | 本工具生成 |
|---|---|---|
| 矩阵类型 | 编译期定维 **Dense**：`Eigen::Matrix<double, 6, 6>` | 运行时定维 **Sparse**：`Eigen::SparseMatrix<double>` |
| 返回形式 | `std::pair<MatrixA, MatrixB>` / `std::tuple<...>` | 单个函数返回稀疏矩阵 |
| 调用方式 | `dynamics_jacobian(x, u)` 成员函数 | 自由函数 |

### 衔接方案（三选一）

**方案 A：生成器增加 Dense 输出模式（推荐）**
给 `generate_function_code` 加一个 `dense=True/False` 参数，输出 `Eigen::Matrix<double, rows, cols>`（编译期定维）。
- 生成结果与 ilqr 的 `MatrixA`/`MatrixB` 类型**完全匹配**，即插即用
- 改动集中在生成器（约 20 行），ilqr 侧零改动或仅改调用名

**方案 B：ilqr 侧加适配层**
在 ilqr 节点里调用生成函数后 `.toDense()` 转换（`Eigen::SparseMatrix` 自带 `toDense()`）。
- 不动生成器，但每次调用有稀疏→稠密转换开销（对 6×6 矩阵可忽略）
- 需要包一层成员函数适配接口

**方案 C：保持现状，仅作参考实现**
生成代码作为"标准答案"用于**测试比对**（数值对照），不替换手写实现。
- 零风险，但失去自动化的意义

> 推荐先做 **方案 A**：稀疏矩阵本质是运行时维度（模板参数是存储序/索引类型，不是行列数），
> 与 ilqr 的编译期 Dense 天然不匹配；让生成器输出 Dense 才是最顺的接法。
> 接好后 ilqr 的 `tools/` 推导脚本可逐步被本工具取代，形成
> 「改模型 → 符号推导 → 自动生成 C++ → 编译测试」闭环。

## 依赖版本

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | ≥3.9 | 开发环境任意 |
| sympy | ≥1.11 | `requirements.txt` |
| Eigen | 3.3.7 | `WORKSPACE`（sha256 锁定） |
| Bazel | 任意（≥5 推荐） | 仅生成的 BUILD 目标需要 |

## 注意事项

- **数值参数会被展开**：`L=3.0, k=0.0003` 会以字面量进生成代码；若想保持符号（如运行时可变），把 L/k 定义为 `symbols('L k')` 并加入函数参数（`variables` 字典）
- 生成代码只含 `#pragma once` 头文件，可直接 `#include`；`BUILD` 规则由脚本自动维护
- 若函数参数需要 `goal/q/r` 等额外向量，在 `variables` 字典里声明即可（见测试示例）
