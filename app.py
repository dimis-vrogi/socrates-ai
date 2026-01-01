import re
import sys
import json
import streamlit as st
from sympy import (
    symbols, Eq, sympify, E, pi, nsimplify,
    solve as sym_solve,
    sin, cos, tan, cot, asin, acos, atan,
    sqrt, log, Interval, Union, oo
)
from sympy.core.traversal import preorder_traversal
from sympy.solvers.inequalities import solve_univariate_inequality

# -------------------------------
# Safe print (for debugging)
# -------------------------------
def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    print(*objects, sep=sep, end=end, file=file)

# -------------------------------
# Format λύσεων
# -------------------------------
def format_solution(expr):
    s = str(expr)
    s = s.replace("pi", "π")
    s = s.replace("E", "e")
    s = s.replace("sqrt", "√")
    if "k" in s:
        s += " , k ∈ ℤ"
    return s

# -------------------------------
# Normalize input
# -------------------------------
def normalize_expr(expr):
    expr = expr.replace("−", "-").replace("—", "-")
    expr = expr.replace("^", "**")
    expr = re.sub(r'\bημ\b', 'sin', expr)
    expr = re.sub(r'\bσυν\b', 'cos', expr)
    expr = re.sub(r'\bεφ\b', 'tan', expr)
    expr = re.sub(r'\bσφ\b', 'cot', expr)
    expr = expr.replace("√", "sqrt")
    expr = re.sub(r'\bln\s*\((.*?)\)', r'log(\1, e)', expr)
    expr = re.sub(r'(\d)([a-zA-Z_])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z_])(\d)', r'\1*\2', expr)
    expr = re.sub(r'\)([a-zA-Z_\d])', r')*\1', expr)
    return expr

# -------------------------------
# Pretty interval
# -------------------------------
def pretty_interval(interval):
    if isinstance(interval, Interval):
        left = "-∞" if interval.start == -oo else str(interval.start)
        right = "+∞" if interval.end == oo else str(interval.end)
        l_bracket = "(" if interval.left_open else "["
        r_bracket = ")" if interval.right_open else "]"
        return f"{l_bracket}{left}, {right}{r_bracket}"
    if isinstance(interval, Union):
        return " ∪ ".join(pretty_interval(arg) for arg in interval.args)
    return str(interval)

# -------------------------------
# Domain for expression
# -------------------------------
def domain_for_expr(expr, var):
    intervals = []
    for sub in preorder_traversal(expr):
        if sub.func == log:
            arg = sub.args[0]
            ineq = arg > 0
            if ineq.has(var):
                sol = solve_univariate_inequality(ineq, var, relational=False)
                intervals.append(sol)
        if sub.func == sqrt:
            arg = sub.args[0]
            ineq = arg >= 0
            if ineq.has(var):
                sol = solve_univariate_inequality(ineq, var, relational=False)
                intervals.append(sol)
        if sub.func == tan:
            arg = sub.args[0]
            ineq = cos(arg) != 0
            if ineq.has(var):
                sol = solve_univariate_inequality(ineq, var, relational=False)
                intervals.append(sol)
        if sub.func == cot:
            arg = sub.args[0]
            ineq = sin(arg) != 0
            if ineq.has(var):
                sol = solve_univariate_inequality(ineq, var, relational=False)
                intervals.append(sol)

    if not intervals:
        return Interval(-oo, oo)
    domain = intervals[0]
    for d in intervals[1:]:
        domain = domain.intersect(d)
    return domain

# -------------------------------
# Solve trig
# -------------------------------
def solve_trig(lhs, rhs, var):
    k = symbols('k', integer=True)
    rhs_val = rhs
    if lhs.func == sin:
        if abs(rhs_val) > 1:
            return []
        a = asin(rhs_val)
        return [2*k*pi + a, 2*k*pi + (pi - a)]
    if lhs.func == cos:
        if abs(rhs_val) > 1:
            return []
        a = acos(rhs_val)
        return [2*k*pi + a, 2*k*pi - a]
    if lhs.func == tan:
        a = atan(rhs_val)
        return [k*pi + a]
    if lhs.func == cot:
        a = atan(1/rhs_val)
        return [k*pi + a]
    return None

# -------------------------------
# Detect chapters
# -------------------------------
def detect_chapter_sections(user_input):
    with open("theory.json", "r", encoding="utf-8") as f:
        theory_data = json.load(f)

    user_input_lower = normalize_expr(user_input).lower()
    matches = []

    if len(user_input.split(",")) > 1:
        for chapter in theory_data["chapters"]:
            if chapter["chapter"] == "Κεφάλαιο 1 – Συστήματα":
                section = chapter["sections"].get("system_equations", {"name": "Συστήματα"})
                matches.append((chapter, section, 1000))

    trig_funcs = ["sin(", "cos(", "tan(", "cot("]
    if any(f in user_input_lower for f in trig_funcs):
        for chapter in theory_data["chapters"]:
            if chapter["chapter"] == "Κεφάλαιο 3 – Τριγωνομετρία":
                section = chapter["sections"].get("trig_equations", {"name": "Βασικές Τριγωνομετρικές Εξισώσεις"})
                matches.append((chapter, section, 100))

    if "^" in user_input_lower:
        for chapter in theory_data["chapters"]:
            if chapter["chapter"] == "Κεφάλαιο 4 – Πολυώνυμα":
                section = chapter["sections"].get("polynomial_equations", {"name": "Πολυωνυμικές Εξισώσεις"})
                matches.append((chapter, section, 100))

    if "log" in user_input_lower:
        for chapter in theory_data["chapters"]:
            if chapter["chapter"] == "Κεφάλαιο 5 – Εκθετική και Λογαριθμική Συνάρτηση":
                section = chapter["sections"].get("exponential_function", {"name": "Λογάριθμοι"})
                matches.append((chapter, section, 100))

    if not matches:
        matches.append(({"chapter":"Παλαιότερη ύλη"}, {"name":"Ύλη προηγούμενων τάξεων"}, 0))

    matches.sort(key=lambda x: x[2], reverse=True)
    return matches

# -------------------------------
# Solve input
# -------------------------------
def solve_input(user_input):
    exprs = [normalize_expr(e.strip()) for e in user_input.split(",")]

    first_expr_str = exprs[0].split("=")[0]
    first_expr = sympify(first_expr_str)
    var = list(first_expr.free_symbols)[0] if first_expr.free_symbols else symbols('x')

    eqs = []
    domain_sets = []

    for expr in exprs:
        if "=" in expr:
            l, r = expr.split("=", 1)
        else:
            l, r = expr, "0"

        lhs = sympify(l)
        rhs = sympify(r)

        trig = solve_trig(lhs, rhs, var)
        if trig is not None:
            sols = [{var: s} for s in trig]
            domain_sets.append(domain_for_expr(lhs, var))
            domain_final = domain_sets[0]
            for d in domain_sets[1:]:
                domain_final = domain_final.intersect(d)
            return sols, f"{var} ∈ {pretty_interval(domain_final)}"

        eqs.append(Eq(lhs, rhs))
        domain_sets.append(domain_for_expr(lhs, var))

    domain_final = domain_sets[0] if domain_sets else Interval(-oo, oo)
    for d in domain_sets[1:]:
        domain_final = domain_final.intersect(d)

    sols = sym_solve(eqs, var)
    formatted_sols = []
    if isinstance(sols, dict):
        formatted_sols = [sols]
    elif isinstance(sols, list):
        for s in sols:
            if isinstance(s, tuple) and len(s) == 1:
                formatted_sols.append({var: nsimplify(s[0])})
            elif isinstance(s, tuple) and len(s) > 1:
                formatted_sols.append({var: s})
            else:
                formatted_sols.append({var: nsimplify(s)})

    return formatted_sols, f"{var} ∈ {pretty_interval(domain_final)}"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Greek Math Solver", layout="centered")
st.title("Επίλυση Εξισώσεων / Συστημάτων")
st.markdown("Γράψε την εξίσωση ή σύστημα:")

user_input = st.text_input("Εξίσωση ή σύστημα:")

if user_input:
    matches = detect_chapter_sections(user_input)
    solutions, domain = solve_input(user_input)

    st.subheader("Πιθανά κεφάλαια/ενότητες:")
    for chap, sect, score in matches[:3]:
        st.write(f"- {chap.get('chapter', chap)} / {sect.get('name', sect)} (συμφωνία: {score})")

    st.subheader("Πεδίο ορισμού:")
    st.write(domain)

    st.subheader("Λύσεις:")
    if not solutions:
        st.write("Αδύνατο")
    else:
        formatted = [{str(k): format_solution(v)} for sol in solutions for k, v in sol.items()]
        st.write(formatted)
