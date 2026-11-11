"""Server-B: general-purpose utility tools (calculator, current time).

Runs as an independent MCP server process over stdio:
    python -m servers.utility_server
Never print() to stdout in this file: stdout is the MCP wire. Log to stderr instead.
"""

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer(
    "utility",
    instructions="General-purpose tools: exact arithmetic and the current date/time.",
)

# ---------------------------------------------------------------------------
# Plain logic (no MCP involved, easy to unit-test)
# ---------------------------------------------------------------------------

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100
MAX_INT_BITS = 10_000

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Symbols LLMs often produce that Python does not accept (escaped to keep this file ASCII).
_NORMALIZE = {
    "×": "*",   # multiplication sign
    "÷": "/",   # division sign
    "（": "(",   # full-width left parenthesis
    "）": ")",   # full-width right parenthesis
    "^": "**",       # caret meaning "power"
}


def _evaluate(node: ast.AST) -> int | float:
    """Walk the parsed expression tree, allowing only numbers and arithmetic."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"Exponent {right} is too large (limit {MAX_EXPONENT}).")
        result = _BINARY_OPS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > MAX_INT_BITS:
            raise ToolError("Intermediate result is too large.")
        return result
    raise ToolError(
        f"Unsupported element {type(node).__name__!r}. "
        "Only numbers, + - * / // % ** and parentheses are allowed."
    )


def evaluate_expression(expression: str) -> str:
    """Safely evaluate an arithmetic expression; raise ToolError on bad input."""
    for symbol, replacement in _NORMALIZE.items():
        expression = expression.replace(symbol, replacement)
    expression = expression.strip()
    if not expression:
        raise ToolError("Expression is empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError(f"Could not parse {expression!r} as an arithmetic expression.") from None
    try:
        result = _evaluate(tree)
    except ZeroDivisionError:
        raise ToolError("Division by zero.") from None
    except OverflowError:
        raise ToolError("Result is too large to represent.") from None
    if isinstance(result, float):
        return f"{result:.12g}"
    return str(result)


def format_now(timezone: str) -> str:
    """Current date, time and weekday in an IANA timezone; raise ToolError if unknown."""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ToolError(
            f"Unknown timezone {timezone!r}. Use an IANA name such as 'Asia/Shanghai'."
        ) from None
    now = datetime.now(tz)
    return f"{now:%Y-%m-%d %H:%M:%S} {now:%A} ({timezone}, UTC{now:%z})"


# ---------------------------------------------------------------------------
# MCP tools: thin wrappers. The docstring is what the LLM reads.
# ---------------------------------------------------------------------------

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly and return the numeric result.

    Use this for ANY math instead of computing in your head.
    Supports + - * / // % ** and parentheses, e.g. "(3 + 5) * 12" or "2 ** 10 / 7".
    """
    return evaluate_expression(expression)


@mcp.tool()
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """Return the current date, time and weekday.

    Use this whenever the answer depends on today's date or the current time.
    `timezone` is an IANA name, e.g. "Asia/Shanghai", "Europe/London", "UTC".
    """
    return format_now(timezone)


if __name__ == "__main__":
    mcp.run()  # no argument = stdio transport, blocks until the client disconnects
