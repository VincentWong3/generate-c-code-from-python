from sympy import Matrix, symbols, cos, sin, tan
import sympy as sp
import re
from string import Template
import os

def parse_expression(expr):
    """
    Recursively parses a SymPy expression to a C++ code string, handling powers and mathematical functions.

    Parameters:
    - expr: SymPy expression.

    Returns:
    - A string representing the C++ code for the expression.
    """
    if isinstance(expr, sp.Add):
        # For addition, parse each argument separately
        return ' + '.join(parse_expression(arg) for arg in expr.args)
    elif isinstance(expr, sp.Mul):
        # For multiplication, parse each argument separately
        return ' * '.join(parse_expression(arg) for arg in expr.args)
    elif isinstance(expr, sp.Pow):
        # For power, convert to std::pow
        base = parse_expression(expr.base)
        exponent = parse_expression(expr.exp)
        return f'std::pow({base}, {exponent})'
    elif isinstance(expr, sp.Function):
        # Handle functions like sin, cos, etc.
        func_name = expr.func.__name__.lower()  # Ensure the function name is in lowercase
        cpp_func_name = f'std::{func_name}'
        args = ', '.join(parse_expression(arg) for arg in expr.args)
        return f'{cpp_func_name}({args})'
    elif expr.is_Symbol:
        # Return symbol names directly
        return str(expr)
    elif expr.is_Number:
        # Return numbers directly
        return str(float(expr))
    else:
        raise ValueError(f"Unhandled expression type: {expr}")


def generate_function_code(expr_matrix, variables, func_name):
    """
    Generates sparse matrix C++ code using Eigen's SparseMatrix.

    Parameters:
    - expr_matrix: Symbolic expression matrix (SymPy Matrix)
    - variables: Dictionary of variables and their dimensions, e.g., {'x': 3, 'u': 2, 'goal': 3}
    - func_name: Function name

    Returns:
    - C++ code string using Eigen's SparseMatrix
    """
    num_rows, num_cols = expr_matrix.shape

    # Build function parameter list
    param_list = ', '.join(
        [f"const Eigen::Matrix<double, {dim}, 1> &{name}" for name, dim in variables.items()]
    )

    # Use string.Template to construct the template string
    sparse_function_template = Template("""
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

Eigen::SparseMatrix<double> ${func_name}(${params}) {
    Eigen::SparseMatrix<double> out(${rows}, ${cols});
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(${nnz});
    ${body}
    out.setFromTriplets(tripletList.begin(), tripletList.end());
    return out;
}
""")

    # Build the tripletList for the sparse matrix
    triplet_entries = []
    for i in range(num_rows):
        for j in range(num_cols):
            expr = expr_matrix[i, j]
            if expr != 0:
                expr_str = parse_expression(expr)

                # Replace variable names with correct indices
                for var_name, _ in variables.items():
                    expr_str = re.sub(rf'\b{var_name}(\d+)\b', lambda m: f"{var_name}[{m.group(1)}]", expr_str)

                triplet_entries.append(f"tripletList.emplace_back({i}, {j}, {expr_str});")

    triplet_code = "\n    ".join(triplet_entries)

    # Use Template to substitute values into the string
    return sparse_function_template.substitute(
        func_name=func_name,
        params=param_list,
        rows=num_rows,
        cols=num_cols,
        nnz=len(triplet_entries),
        body=triplet_code
    )

def save_code_to_file(cpp_code, header_filename, directory="c_generated_code"):
    """
    Save the generated C++ code to a file and update the BUILD file.

    Parameters:
    - cpp_code: C++ code as a string.
    - header_filename: Name of the header file to include in the BUILD file.
    - directory: Directory where the file and BUILD file will be saved.
    """
    # Construct the path to the parent directory
    parent_directory = os.path.join(os.getcwd(), os.path.pardir, directory)

    # Ensure the directory exists
    os.makedirs(parent_directory, exist_ok=True)

    # Save the C++ code to a .cpp file
    cpp_filepath = os.path.join(parent_directory, header_filename)
    with open(cpp_filepath, 'w') as file:
        file.write(cpp_code)
    print(f"C++ code saved to {cpp_filepath}")

    # Create or update the Bazel BUILD file
    build_filepath = os.path.join(parent_directory, "BUILD")
    rule_name = header_filename.replace('.h', '')
    cc_library_rule = f"""
cc_library(
    name = "{rule_name}",
    hdrs = [
        "{header_filename}",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)
"""

    # Check if BUILD file exists and if the rule already exists
    if os.path.exists(build_filepath):
        with open(build_filepath, 'r') as build_file:
            build_content = build_file.read()
            # Use regex to match the exact rule name in the name field
            if f'name = "{rule_name}"' in build_content:
                print(f"Rule with name '{rule_name}' already exists in BUILD file. Skipping update.")
                return

        # Append to existing BUILD file if rule does not exist
        with open(build_filepath, 'a') as build_file:
            build_file.write(cc_library_rule)
        print(f"Updated BUILD file at {build_filepath}")
    else:
        # Create new BUILD file if it does not exist
        with open(build_filepath, 'w') as build_file:
            build_file.write(cc_library_rule)
        print(f"Created new BUILD file at {build_filepath}")

