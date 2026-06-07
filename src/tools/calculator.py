import logging
import ast
import math
import operator
import re

logger = logging.getLogger(__name__)

ALLOWED_OPERATORS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(_safe_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression: {node}")


def _extract_expression(text: str) -> str:
    """Extract math expression from natural language"""

    text = text.lower().strip()

    # Handle "X% of Y"
    pct_match = re.search(r"([\d.]+)\s*%\s*of\s*([\d.]+)", text)
    if pct_match:
        pct = float(pct_match.group(1))
        val = float(pct_match.group(2))
        return str(pct / 100 * val)

    # Handle "sqrt of X" or "sqrt(X)"
    sqrt_match = re.search(r"sqrt[\s(]*([\d.]+)", text)
    if sqrt_match:
        num = float(sqrt_match.group(1))
        return str(math.sqrt(num))

    # Extract just the math expression using regex
    expr_match = re.search(
        r"[\d\s\+\-\*\/\(\)\.\^]+", text
    )
    if expr_match:
        return expr_match.group(0).strip()

    return text


def calculator(expression: str) -> str:
    try:
        logger.info(f"[TOOL] calculator input: {expression}")

        clean_expr = _extract_expression(expression)
        logger.info(f"[TOOL] calculator expr: '{clean_expr}'")

        # Guard against empty expression
        if not clean_expr or not clean_expr.strip():
            return "Could not find a math expression to calculate."

        try:
            result = float(clean_expr)
        except ValueError:
            tree   = ast.parse(clean_expr.strip(), mode="eval")
            result = _safe_eval(tree.body)

        result_str = str(round(result, 6))
        logger.info(f"[TOOL] calculator result: {result_str}")
        return result_str

    except Exception as e:
        logger.error(f"[TOOL] calculator error: {e}")
        return f"Could not calculate: {str(e)}"
