import streamlit as st
import os
import random
import math
import re
import google.generativeai as genai


# ==========================================
# 1. EML 神经符号引擎 (内核：无跳步绝对严格拓扑推演)
# ==========================================
class Expr:
    def __add__(self, other): return Add(self, _to_expr(other))
    def __radd__(self, other): return Add(_to_expr(other), self)
    def __sub__(self, other): return Add(self, Mul(Const(-1), _to_expr(other)))
    def __rsub__(self, other): return Add(_to_expr(other), Mul(Const(-1), self))
    def __mul__(self, other): return Mul(self, _to_expr(other))
    def __rmul__(self, other): return Mul(_to_expr(other), self)
    def __truediv__(self, other): return Div(self, _to_expr(other))
    def __rtruediv__(self, other): return Div(_to_expr(other), self)
    def __pow__(self, other): return Pow(self, _to_expr(other))
    def __rpow__(self, other): return Pow(_to_expr(other), self)
    def __neg__(self): return Mul(Const(-1), self)
    def deriv(self, var): raise NotImplementedError


def _to_expr(val):
    if isinstance(val, Expr): return val
    return Const(val)


class Const(Expr):
    def __init__(self, val): self.val = val
    def deriv(self, var):
        return Const(0), [f"$$ \\frac{{d}}{{d{var}}}({self.val}) = 0 $$"]
    def eval(self, env): return self.val
    def __str__(self): return str(self.val)


class Var(Expr):
    def __init__(self, name): self.name = name
    def deriv(self, var):
        if self.name == var:
            return Const(1), [f"$$ \\frac{{d}}{{d{var}}}({self.name}) = 1 $$"]
        return Const(0), [f"$$ \\frac{{d}}{{d{var}}}({self.name}) = 0 $$"]
    def eval(self, env): return env.get(self.name, 0)
    def __str__(self): return self.name


class Add(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        return ld + rd, ls + rs + [
            f"$$ \\frac{{d}}{{d{var}}}({self.left} + {self.right}) = \\frac{{d}}{{d{var}}}({self.left}) + \\frac{{d}}{{d{var}}}({self.right}) = {ld} + {rd} $$"]
    def eval(self, env): return self.left.eval(env) + self.right.eval(env)
    def __str__(self): return f"({self.left} + {self.right})"


class Mul(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        res = ld * self.right + self.left * rd
        return res, ls + rs + [
            f"$$ \\frac{{d}}{{d{var}}}({self.left} \\cdot {self.right}) = \\frac{{d}}{{d{var}}}({self.left}) \\cdot {self.right} + {self.left} \\cdot \\frac{{d}}{{d{var}}}({self.right}) = ({ld})({self.right}) + ({self.left})({rd}) $$"]
    def eval(self, env): return self.left.eval(env) * self.right.eval(env)
    def __str__(self): return f"({self.left} \\cdot {self.right})"


class Pow(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        if isinstance(self.right, Const):
            n = self.right.val
            ld, ls = self.left.deriv(var)
            res = Const(n) * (self.left ** Const(n - 1)) * ld
            step = f"$$ \\frac{{d}}{{d{var}}}({self.left}^{{{n}}}) = {n}{self.left}^{{{n}-1}} \\cdot \\frac{{d}}{{d{var}}}({self.left}) = {n}{self.left}^{{{n}-1}} \\cdot ({ld}) $$"
            return res, ls + [step]
        return Const(0), ["目前引擎仅支持常数幂拓扑"]
    def eval(self, env): return self.left.eval(env) ** self.right.eval(env)
    def __str__(self): return f"({self.left})^{{{self.right}}}"


class Div(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        res = (ld * self.right - self.left * rd) / (self.right * self.right)
        step = f"$$ \\frac{{d}}{{d{var}}}\\left(\\frac{{{self.left}}}{{{self.right}}}\\right) = \\frac{{\\frac{{d}}{{d{var}}}({self.left}) \\cdot {self.right} - {self.left} \\cdot \\frac{{d}}{{d{var}}}({self.right})}}{{{self.right}^2}} = \\frac{{({ld}){self.right} - {self.left}({rd})}}{{{self.right}^2}} $$"
        return res, ls + rs + [step]
    def eval(self, env): return self.left.eval(env) / self.right.eval(env)
    def __str__(self): return f"\\frac{{{self.left}}}{{{self.right}}}"


# ==========================================
# 核心 EML 算子
# ==========================================
class Eml(Expr):
    def __init__(self, x, y):
        self.x = _to_expr(x)
        self.y = _to_expr(y)

    def deriv(self, var):
        xd, xs = self.x.deriv(var)
        yd, ys = self.y.deriv(var)

        term1 = eml(self.x, 1) * xd
        term2 = (Const(1) / self.y) * yd
        res = term1 - term2

        step = f"$$ \\frac{{d}}{{d{var}}}eml({self.x}, {self.y}) = eml({self.x},1)\\cdot{xd} - \\frac{{1}}{{{self.y}}}\\cdot{yd} $$"
        return res, xs + ys + [step]

    def eval(self, env):
        return math.exp(self.x.eval(env)) - math.log(self.y.eval(env))

    def __str__(self):
        return f"eml({self.x},{self.y})"


# 外部宏函数（正确写在类外面）
def eml(x, y): return Eml(x, y)
def exp(x): return eml(x, 1)
def ln(x): return eml(1, eml(eml(1, x), 1))


# ==========================================
# 2. UI 界面
# ==========================================
st.set_page_config(page_title="EML 符号证明器", page_icon="🧪")
st.title("🧪 EML 神经符号证明引擎")

with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("输入 Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))


# ==========================================
# 3. 证明核心流程
# ==========================================
if api_key:
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected_model_name = next((name for name in available_models if "flash" in name), available_models[0])
        model = genai.GenerativeModel(selected_model_name)
        st.sidebar.success(f"✅ 核心已连接: {selected_model_name}")
    except Exception as e:
        st.error(f"❌ 访问 API 失败：{str(e)}")
        st.stop()

    user_input = st.text_input("输入你想证明的问题：", placeholder="例如：证明 ln(x^2) 的导数等于 2/x")

    if st.button("启动 EML 符号证明"):
        if not user_input:
            st.warning("请输入问题后再试。")
        else:
            with st.spinner("EML 引擎正在解析语义并构建拓扑树..."):
                try:
                    prompt = f"""
                    You are a strict math parser for an EML engine.
                    Extract ONLY the original mathematical expression.
                    Do NOT output results or Chinese.
                    Use ** for power.
                    Allowed: x, y, ln, exp.

                    Example 1: 证明 ln(x^2) 导数是 2/x → ln(x**2)
                    Example 2: 求 x*exp(x) 导数 → x*exp(x)
                    Example 3: 1/x 导数 → 1/x

                    User: {user_input}
                    Output only expression:
                    """

                    raw_text = model.generate_content(prompt).text.strip()

                    # 清洗
                    clean_code = raw_text
                    clean_code = re.sub(r'span_\d+', '', clean_code)
                    clean_code = re.sub(r"'", '', clean_code)
                    clean_code = clean_code.replace('```python', '').replace('```', '').replace('`', '').strip()
                    clean_code = clean_code.replace('，', ',').replace('（', '(').replace('）', ')')
                    clean_code = clean_code.replace('^', '**')
                    clean_code = clean_code.replace('math.log', 'ln').replace('np.log', 'ln')
                    clean_code = clean_code.replace('math.exp', 'exp').replace('np.exp', 'exp')
                    if clean_code.lower().startswith('python'):
                        clean_code = clean_code[6:].strip()

                    st.info(f"🔍 EML 提取结构: `{clean_code}`")

                    # 执行
                    env_vars = {'x': Var('x'), 'y': Var('y'), 'ln': ln, 'exp': exp, 'eml': eml}
                    lhs_expr = eval(clean_code, {"__builtins__": {}}, env_vars)
                    derived_expr, steps = lhs_expr.deriv('x')

                    st.markdown("### 📜 EML 推导过程")
                    for step in steps:
                        st.write(step)

                    st.markdown("---")
                    st.markdown("### 🏁 最终结论")
                    st.latex(f"\\frac{{d}}{{dx}}({lhs_expr}) = {derived_expr}")

                except Exception as e:
                    st.error(f"EML 错误：{type(e).__name__} - {str(e)}")
