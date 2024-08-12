from sympy import Matrix, symbols
import re

def calculate_dynamics_and_derivatives(x_dim, u_dim, f_vector):
    """
    计算动力学方程及其导数。

    参数：
    - x_dim: 状态变量的数量
    - u_dim: 控制变量的数量
    - f_vector: 状态方程向量（SymPy 矩阵）

    返回：
    - 显式雅可比矩阵、隐式雅可比矩阵
    """
    # 定义状态变量、控制变量和 x_dot 符号
    x = Matrix(symbols(f'x:{x_dim}'))
    u = Matrix(symbols(f'u:{u_dim}'))
    x_dot = Matrix(symbols(f'x_dot:{x_dim}'))

    # 计算显式雅可比矩阵
    jacobian_x = f_vector.jacobian(x)
    jacobian_u = f_vector.jacobian(u)

    # 定义隐式方程 f_implicit = x_dot - f(x, u)
    f_implicit = x_dot - f_vector

    # 计算隐式雅可比矩阵
    jacobian_implicit_x = f_implicit.jacobian(x)
    jacobian_implicit_u = f_implicit.jacobian(u)
    jacobian_implicit_x_dot = f_implicit.jacobian(x_dot)

    return jacobian_x, jacobian_u, jacobian_implicit_x, jacobian_implicit_u, jacobian_implicit_x_dot

def calculate_cost_and_derivatives(x_dim, u_dim, cost_expr):
    """
    计算成本函数及其雅可比和 Hessian 矩阵。

    参数：
    - x_dim: 状态变量的数量
    - u_dim: 控制变量的数量
    - goal_dim: 目标变量的数量
    - cost_expr: 成本函数的符号表达式（SymPy 符号）

    返回：
    - 成本函数、雅可比和 Hessian 矩阵的符号表达式
    """
    # 定义状态变量、控制变量和目标符号
    x = Matrix(symbols(f'x:{x_dim}'))
    u = Matrix(symbols(f'u:{u_dim}'))

    # 计算雅可比矩阵
    jacobian_x = Matrix([cost_expr.diff(var) for var in x]).reshape(1, x_dim)
    jacobian_u = Matrix([cost_expr.diff(var) for var in u]).reshape(1, u_dim)

    # 计算 Hessian 矩阵
    hessian_xx = jacobian_x.jacobian(x)
    hessian_uu = jacobian_u.jacobian(u)
    hessian_ux = jacobian_u.jacobian(x)

    return jacobian_x, jacobian_u, hessian_xx, hessian_uu, hessian_ux