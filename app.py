import streamlit as st
import os
import random
import math
import re
import google.generativeai as genai

# ==========================================
# 1. EML 神经符号引擎 (内核：魔法方法全量补齐)
# ==========================================
class Expr:
    def __add__(self, other): return Add(self, _to_expr(other))
    def __radd__(self, other): return Add(_to_expr(other), self)
    def __sub__(self, other): return Add(self, Mul(Const(-1), _to_expr(other)))
    def __rsub__(self, other): return Add(_to_expr(other), Mul(Const(-1), self))  # 优化：反向减法逻辑
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
    def deriv(self, var): return Const(0), [f"常数规则: $d({self.val})/d{var} = 0$"]
    def eval(self, env): return self.val
    def __str__(self): return str(self.val)

class Var(Expr):
    def __init__(self, name): self.name = name
    def deriv(self, var):
        if self.name == var: return Const(1), [f"自变量基础: $d({self.name})/d{self.name} = 1$"]
        return Const(0), [f"偏导规则: $d({self.name})/d{var} = 0$"]
    def eval(self, env): return env.get(self.name, 0)
    def __str__(self): return self.name

class Add(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        return ld + rd, ls + rs + [f"加法规则: $d({self.left} + {self.right}) = d({self.left}) + d({self.right})$"]
    def eval(self, env): return self.left.eval(env) + self.right.eval(env)
    def __str__(self): return f"({self.left} + {self.right})"

class Mul(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        res = ld * self.right + self.left * rd
        return res, ls + rs + [f"乘法法则(Leibniz): $d({self.left} \\cdot {self.right}) = ({ld})({self.right}) + ({self.left})({rd})$"]
    def eval(self, env): return self.left.eval(env) * self.right.eval(env)
    def __str__(self): return f"({self.left} \\cdot {self.right})"

class Pow(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        if isinstance(self.right, Const):
            n = self.right.val
            ld, ls = self.left.deriv(var)
            res = Const(n) * (self.left ** Const(n-1)) * ld
            return res, ls + [f"幂链式法则: $d(u^{{{n}}}) = {n}u^{{{n-1}}} \\cdot du$"]
        return Const(0), ["目前引擎仅支持常数幂求导"]
    def eval(self, env): return self.left.eval(env) ** self.right.eval(env)
    def __str__(self): return f"({self.left})^{{{self.right}}}"

class Exp(Expr):
    def __init__(self, inner): self.inner = inner
    def deriv(self, var):
        id_res, id_steps = self.inner.deriv(var)
        res = exp(self.inner) * id_res
        return res, id_steps + [f"指数规则: $d(e^{{{self.inner}}}) = e^{{{self.inner}}} \\cdot d({self.inner})$"]
    def eval(self, env): return math.exp(self.inner.eval(env))
    def __str__(self): return f"e^{{{self.inner}}}"

def exp(x): return Exp(_to_expr(x))

class Ln(Expr):
    def __init__(self, inner): self.inner = inner
    def deriv(self, var):
        id_res, id_steps = self.inner.deriv(var)
        res = id_res / self.inner
        return res, id_steps + [f"对数链式法则: $d(\\ln({self.inner})) = \\frac{{1}}{{{self.inner}}} \\cdot d({self.inner}) = {res}$"]
    def eval(self, env): return math.log(self.inner.eval(env))
    def __str__(self): return f"\\ln({self.inner})"

class Div(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        res = (ld * self.right - self.left * rd) / (self.right * self.right)
        return res, ls + rs + [f"商法则: $d({self.left}/{self.right}) = \\frac{{u'v - uv'}}{{v^2}}$"]
    def eval(self, env): return self.left.eval(env) / self.right.eval(env)
    def __str__(self): return f"\\frac{{{self.left}}}{{{self.right}}}"

def ln(x): return Ln(_to_expr(x))

# ==========================================
# 2. UI 界面设计
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
        st.sidebar.success(f"✅ 已连接: {selected_model_name}")
    except Exception as e:
        st.error(f"❌ 访问 API 失败：{str(e)}")
        st.stop()

    user_input = st.text_input("输入你想证明的问题：", placeholder="例如：证明 ln(x^2) 的导数")

    if st.button("执行 EML 符号证明"):
        if user_input:
            with st.spinner("AI 正在解析语义并构建 EML 拓扑树..."):
                try:
                    # A: 语义编译 (只要左边表达式，不要等号)
                    prompt = (
                        "Translate the math problem into a Python expression for the LHS ONLY. "
                        "Variables: x, y. Functions: ln, exp. Output ONLY the expression, "
                        f"no backticks, no equations. Problem: {user_input}"
                    )
                    raw_text = model.generate_content(prompt).text.strip()

                    # 强力清洗（修复：正则表达式语法错误）
                    clean_code = re.sub(r'span_\d+', '', raw_text)
                    clean_code = re.sub(r"'", '', clean_code)  # 修复：正确去除单引号
                    clean_code = clean_code.replace('```python', '').replace('```', '').replace('`', '').strip()
                    clean_code = clean_code.replace('，', ',').replace('（', '(').replace('）', ')')
                    clean_code = clean_code.replace('^', '**')
                    if clean_code.lower().startswith('python'): clean_code = clean_code[6:].strip()

                    # 🔍 增加了一个“透视镜”，让你看到清洗后的代码
                    st.info(f"🔍 AI 编译出的表达式: `{clean_code}`")

                    # B: 符号推导
                    env_vars = {'x': Var('x'), 'y': Var('y'), 'ln': ln, 'exp': exp}

                    # 执行符号转换
                    lhs_expr = eval(clean_code, {"__builtins__": {}}, env_vars)

                    derived_expr, steps = lhs_expr.deriv('x')

                    st.markdown("### 📜 EML 符号推导过程")
                    for i, step in enumerate(steps):
                        st.success(f"**Step {i + 1}**: {step}")

                    st.markdown("---")
                    st.markdown("### 🏁 最终证明结论")
                    st.latex(f"\\frac{{d}}{{dx}}({lhs_expr}) = {derived_expr}")

                except Exception as e:
                    # 如果再报错，这里会打印出真正的元凶（比如 NameError 或 SyntaxError）
                    st.error(f"证明出错：{type(e).__name__} - {str(e)}")
        else:
            st.warning("请输入问题后再试。")
