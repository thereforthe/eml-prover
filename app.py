import streamlit as st
import os
import random
import math
import re
import google.generativeai as genai

# ==========================================
# 1. EML 神经符号引擎 (内核：深度融合 EML 拓扑推演)
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
        return Const(0), [f"⬛ [EML 终结节点] 常数 {self.val} 拓扑坍缩: $\\frac{{d}}{{d{var}}}({self.val}) \\Rightarrow 0$"]
    def eval(self, env): return self.val
    def __str__(self): return str(self.val)

class Var(Expr):
    def __init__(self, name): self.name = name
    def deriv(self, var):
        if self.name == var:
            return Const(1), [f"🟩 [EML 根节点] 自变量映射激活: $\\frac{{d}}{{d{var}}}({self.name}) \\Rightarrow 1$"]
        return Const(0), [f"⬛ [EML 偏导节点] 无关变量坍缩: $\\frac{{d}}{{d{var}}}({self.name}) \\Rightarrow 0$"]
    def eval(self, env): return env.get(self.name, 0)
    def __str__(self): return self.name

class Add(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        return ld + rd, ls + rs + [f"💠 [EML 拓扑合并 | 线性加法] 节点分裂: $\\frac{{d}}{{d{var}}}({self.left} + {self.right}) \\Rightarrow {ld} + {rd}$"]
    def eval(self, env): return self.left.eval(env) + self.right.eval(env)
    def __str__(self): return f"({self.left} + {self.right})"

class Mul(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        res = ld * self.right + self.left * rd
        return res, ls + rs + [f"💠 [EML 拓扑合并 | 乘法交叉] 莱布尼茨网络展开: $\\frac{{d}}{{d{var}}}({self.left} \\cdot {self.right}) \\Rightarrow ({ld})({self.right}) + ({self.left})({rd})$"]
    def eval(self, env): return self.left.eval(env) * self.right.eval(env)
    def __str__(self): return f"({self.left} \\cdot {self.right})"

class Pow(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        if isinstance(self.right, Const):
            n = self.right.val
            ld, ls = self.left.deriv(var)
            res = Const(n) * (self.left ** Const(n-1)) * ld
            return res, ls + [f"🌀 [EML 拓扑代换 | 幂运算] 链式降维打击: $\\frac{{d}}{{d{var}}}({self.left}^{{{n}}}) \\Rightarrow {n}{self.left}^{{{n-1}}} \\otimes ({ld})$"]
        return Const(0), ["目前 EML 引擎仅支持常数幂拓扑"]
    def eval(self, env): return self.left.eval(env) ** self.right.eval(env)
    def __str__(self): return f"({self.left})^{{{self.right}}}"

class Exp(Expr):
    def __init__(self, inner): self.inner = inner
    def deriv(self, var):
        id_res, id_steps = self.inner.deriv(var)
        res = exp(self.inner) * id_res
        return res, id_steps + [f"🌀 [EML 拓扑代换 | 指数] 神经同构映射: $\\frac{{d}}{{d{var}}}(e^{{{self.inner}}}) \\Rightarrow e^{{{self.inner}}} \\otimes ({id_res})$"]
    def eval(self, env): return math.exp(self.inner.eval(env))
    def __str__(self): return f"e^{{{self.inner}}}"

def exp(x): return Exp(_to_expr(x))

class Ln(Expr):
    def __init__(self, inner): self.inner = inner
    def deriv(self, var):
        id_res, id_steps = self.inner.deriv(var)
        res = id_res / self.inner
        return res, id_steps + [f"🌀 [EML 拓扑代换 | 对数] 倒数结构生成: $\\frac{{d}}{{d{var}}}(\\ln({self.inner})) \\Rightarrow \\frac{{1}}{{{self.inner}}} \\otimes ({id_res})$"]
    def eval(self, env): return math.log(self.inner.eval(env))
    def __str__(self): return f"\\ln({self.inner})"

class Div(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        res = (ld * self.right - self.left * rd) / (self.right * self.right)
        return res, ls + rs + [f"💠 [EML 拓扑合并 | 商空间] 分式展开: $\\frac{{d}}{{d{var}}}(\\frac{{{self.left}}}{{{self.right}}}) \\Rightarrow \\frac{{({ld}){self.right} - {self.left}({rd})}}{{{self.right}^2}}$"]
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
        st.sidebar.success(f"✅ 核心已连接: {selected_model_name}")
    except Exception as e:
        st.error(f"❌ 访问 API 失败：{str(e)}")
        st.stop()

    user_input = st.text_input("输入你想证明的问题：", placeholder="例如：证明 ln(x^2) 的导数等于 2/x")

    if st.button("启动 EML 符号证明"):
        if user_input:
            with st.spinner("EML 引擎正在解析语义并构建拓扑树..."):
                try:
                    # A: 语义编译 (严格 Few-Shot 提示，杜绝提取答案)
                    prompt = f"""
                    You are a strict math parser for an EML engine. 
                    Extract ONLY the original mathematical expression that needs to be differentiated from the user's problem.
                    Do NOT output the expected result (e.g. 2/x), the equation, or any Chinese words. 
                    Convert to valid Python code (use ** for power).
                    Variables allowed: x, y. Functions allowed: ln, exp.

                    Example 1: "证明 ln(x^2) 的导数等于 2/x" -> ln(x**2)
                    Example 2: "求 x*exp(x) 的导数" -> x*exp(x)
                    Example 3: "证明 1/x 的导数是 -1/x^2" -> 1/x

                    User Problem: {user_input}
                    Output ONLY the python expression:
                    """

                    raw_text = model.generate_content(prompt).text.strip()

                    # B: 强力清洗层
                    clean_code = raw_text  # 确保变量初始化！
                    clean_code = re.sub(r'span_\d+', '', clean_code)
                    clean_code = re.sub(r"'", '', clean_code)  # 修复：正则表达式语法错误
                    clean_code = clean_code.replace('```python', '').replace('```', '').replace('`', '').strip()
                    clean_code = clean_code.replace('，', ',').replace('（', '(').replace('）', ')')
                    clean_code = clean_code.replace('^', '**')
                    clean_code = clean_code.replace('math.log', 'ln').replace('np.log', 'ln')
                    clean_code = clean_code.replace('math.exp', 'exp').replace('np.exp', 'exp')
                    if clean_code.lower().startswith('python'): clean_code = clean_code[6:].strip()

                    st.info(f"🔍 EML 预编译器提取结构: `{clean_code}`")

                    # C: EML 符号推导核心
                    env_vars = {'x': Var('x'), 'y': Var('y'), 'ln': ln, 'exp': exp}

                    # 生成拓扑树
                    lhs_expr = eval(clean_code, {"__builtins__": {}}, env_vars)

                    # 执行符号运算
                    derived_expr, steps = lhs_expr.deriv('x')

                    st.markdown("### 📜 EML 神经符号推导流")
                    for i, step in enumerate(steps):
                        st.success(f"**操作 {i + 1}**: {step}")

                    st.markdown("---")
                    st.markdown("### 🏁 EML 最终状态方程")
                    st.latex(f"\\frac{{d}}{{dx}}({lhs_expr}) = {derived_expr}")

                except Exception as e:
                    st.error(f"EML 引擎执行中断：{type(e).__name__} - {str(e)}")
        else:
            st.warning("请输入问题后再试。")
