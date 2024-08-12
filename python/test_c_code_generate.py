from sympy import Matrix, symbols, exp, log, sin, cos, tan, asin
from generate_function_c_code import *
from calc_jacobian_and_hessian import *

def run_tests():
    # Define the dimensions
    x_dim = 6
    u_dim = 2
    x = Matrix(symbols(f'x:{x_dim}'))
    u = Matrix(symbols(f'u:{u_dim}'))
    x_dot = Matrix(symbols(f'x_dot:{x_dim}'))
    goal = Matrix(symbols(f'goal:{x_dim}'))
    q = Matrix(symbols(f'q:{x_dim}'))
    r = Matrix(symbols(f'r:{u_dim}'))



    variables_x_u = {'x': x_dim, 'u': u_dim}
    variables_x_u_goal = {'x': x_dim, 'u': u_dim, 'goal': x_dim}
    variables_x_u_goal_q_r = {'x': x_dim, 'u': u_dim, 'goal': x_dim, 'q': x_dim, 'r': u_dim}
    variables_x_u_x_dot = {'x': x_dim, 'u': u_dim, 'x_dot': x_dim}

    error = x - goal
    cost_state = sum(error[i] * q[i] * error[i] for i in range(x_dim))
    cost_control = sum(u[i] * r[i] * u[i] for i in range(u_dim))
    cost_expr = Matrix([[cost_state + cost_control]])

    L = 3.0
    k = 0.0003

    f0 = x[4] * cos(x[2])
    f1 = x[4] * sin(x[2])
    f2 = x[4] * tan(x[3]) / (L * (1 + k * x[4] * x[4]))
    f3 = u[0]
    f4 = x[5]
    f5 = u[1]
    fxu = Matrix([f0, f1, f2, f3, f4, f5])
    imp_fxu = x_dot - fxu



    cost_jacobian_x, cost_jacobian_u, cost_hessian_xx, cost_hessian_uu, cost_hessian_ux = (
        calculate_cost_and_derivatives(x_dim, u_dim,cost_expr))

    dynamic_jacobian_x, dynamic_jacobian_u, dynamic_jacobian_implicit_x, dynamic_jacobian_implicit_u, dynamic_jacobian_implicit_x_dot = (
        calculate_dynamics_and_derivatives(x_dim, u_dim, fxu))

    dynamic_continuous_code = generate_function_code(fxu, variables_x_u, f"dynamic_continuous_code")
    print(dynamic_continuous_code)
    cost_code = generate_function_code(cost_expr, variables_x_u_goal_q_r, f"cost")
    print(cost_code)
    cost_jacobian_x_code = generate_function_code(cost_jacobian_x, variables_x_u_goal_q_r, f"cost_jacobian_x")
    print(cost_jacobian_x_code)
    cost_jacobian_u_code = generate_function_code(cost_jacobian_u, variables_x_u_goal_q_r, f"cost_jacobian_u")
    print(cost_jacobian_u_code)
    cost_hessian_xx_code = generate_function_code(cost_hessian_xx, variables_x_u_goal_q_r, f"cost_hessian_xx")
    print(cost_hessian_xx_code)
    cost_hessian_uu_code = generate_function_code(cost_hessian_uu, variables_x_u_goal_q_r, f"cost_hessian_uu")
    print(cost_hessian_uu_code)
    cost_hessian_ux_code = generate_function_code(cost_hessian_ux, variables_x_u_goal_q_r, f"cost_hessian_ux")
    print(cost_hessian_ux_code)
    dynamic_jacobian_x_code = generate_function_code(dynamic_jacobian_x, variables_x_u,f"dynamic_jacobian_x")
    print(dynamic_jacobian_x_code)
    dynamic_jacobian_u_code = generate_function_code(dynamic_jacobian_u, variables_x_u, f"dynamic_jacobian_u")
    print(dynamic_jacobian_u_code)
    save_code_to_file(dynamic_continuous_code, f"dynamic_continuous.h")
    save_code_to_file(dynamic_jacobian_x_code, f"dynamic_jacobian_x.h")
    save_code_to_file(dynamic_jacobian_u_code, f"dynamic_jacobian_u.h")
    save_code_to_file(cost_code, f"cost.h")
    save_code_to_file(cost_jacobian_x_code, f"cost_jacobian_x.h")
    save_code_to_file(cost_jacobian_u_code, f"cost_jacobian_u.h")
    save_code_to_file(cost_hessian_xx_code, f"cost_hessian_xx.h")
    save_code_to_file(cost_hessian_uu_code, f"cost_hessian_uu.h")



run_tests()
