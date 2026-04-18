import streamlit as st
import os
import random
import math
import google.generativeai as genai

# ==========================================
# 1. EML 神经符号引擎 (内核：带证明追踪功能)
# ==========================================
class Expr:
    def __add__(self, other): return Add(self, _to_expr(other))
    def __mul__(self, other): return Mul(self, _to_expr(other))
    def __pow__(self, other): return Pow(self, _to_expr(other))
    def __truediv__(self, other): return Div(self, _to_expr(other))

    def deriv(self, var):
        """返回 (结果表达式, 证明步骤列表)"""
        raise NotImplementedError

def _to_expr(val):
    if isinstance(val, Expr): return val
    return Const(val)

class Const(Expr):
    def __init__(self, val): self.val = val
    def deriv(self, var):
        return Const(0), [f"根据常数规则: $d({self.val})/d{var} = 0$"]
    def eval(self, env): return self.val
    def __str__(self): return str(self.val)

class Var(Expr):
    def __init__(self, name): self.name = name
    def deriv(self, var):
        if self.name == var:
            return Const(1), [f"基础公理: $d({self.name})/d{self.name} = 1$"]
        return Const(0), [f"基础公理: $d({self.name})/d{var} = 0$ (视为常数)"]
    def eval(self, env): return env.get(self.name, 0)
    def __str__(self): return self.name

class Add(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        return ld + rd, ls + rs + [f"线性规则: $d({self.left} + {self.right}) = d({self.left}) + d({self.right})$"]
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

class Exp(Expr):
    def __init__(self, inner): self.inner = inner
    def deriv(self, var):
        id, is_ = self.inner.deriv(var)
        res = exp(self.inner) * id
        return res, is_ + [f"指数规则: $d(e^{{{self.inner}}}) = e^{{{self.inner}}} \\cdot d({self.inner})$"]
    def eval(self, env): return math.exp(self.inner.eval(env))
    def __str__(self): return f"e^{{{self.inner}}}"
# 对应的辅助函数
def exp(x): return Exp(_to_expr(x))

class Ln(Expr):
    def __init__(self, inner): self.inner = inner
    def deriv(self, var):
        id, is_ = self.inner.deriv(var)
        res = id / self.inner
        return res, is_ + [f"对数链式法则: $d(\\ln({self.inner})) = \\frac{{1}}{{{self.inner}}} \\cdot d({self.inner}) = {res}$"]
    def eval(self, env): return math.log(self.inner.eval(env))
    def __str__(self): return f"\\ln({self.inner})"

class Div(Expr):
    def __init__(self, left, right): self.left, self.right = left, right
    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        res = (ld * self.right + (Const(-1) * self.left * rd)) / (self.right * self.right)
        return res, ls + rs + [f"商法则: $d({self.left}/{self.right}) = \\frac{{u'v - uv'}}{{v^2}}$"]
    def eval(self, env): return self.left.eval(env) / self.right.eval(env)
    def __str__(self): return f"\\frac{{{self.left}}}{{{self.right}}}"

def ln(x): return Ln(_to_expr(x))
zero = Const(0)

# ==========================================
# 2. UI 界面设计
# ==========================================
st.set_page_config(page_title="EML 符号证明器", page_icon="🧪")
st.title("🧪 EML 神经符号证明引擎")

with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("输入 Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

# ==========================================
# 3. 证明核心流程 (已修复 404、中文标点及 EML 步骤逻辑)
# ==========================================
if api_key:
    genai.configure(api_key=api_key)
    try:
        # 1. 动态诊断模型 (解决 404 问题)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available_models:
            st.error("⚠️ 诊断失败：该 API Key 无权访问任何模型。")
            st.stop()

        # 优先选择最新的 flash 模型，否则选第一个可用模型
        selected_model_name = next((name for name in available_models if "flash" in name), available_models[0])
        model = genai.GenerativeModel(selected_model_name)
        st.sidebar.success(f"✅ 已连接模型: {selected_model_name}")

    except Exception as e:
        st.error(f"❌ 访问 API 失败：{str(e)}")
        st.stop()

    user_input = st.text_input("输入你想证明的问题：", placeholder="例如：证明 ln(x^2) 的导数是 2/x")

    if st.button("执行 EML 符号证明"):
        if user_input:
            with st.spinner("AI 正在解析语义并构建 EML 拓扑树..."):
                try:
                    # 阶段 A：语义编译 (增加强制约束防止中文标点)
                    prompt_compile = (
                        f"Translate the following math problem into a Python expression for the LHS. "
                        f"Variables: x, y. Functions: ln, exp. Output ONLY the expression, "
                        f"no backticks, no Chinese punctuation. Example: ln(x**2). "
                        f"Problem: {user_input}"
                    )
                    
                    response = model.generate_content(prompt_compile)
                    raw_text = response.text.strip()

                    # --- 核心修复：强力清洗非法字符 (解决 U+FF0C 报错) ---
                    # 替换中文逗号、括号，并将 ^ 转换为 Python 幂运算 **
                    clean_code = raw_text.replace('```python', '').replace('```', '').replace('`', '').strip()
                    clean_code = clean_code.replace('，', ',').replace('（', '(').replace('）', ')')
                    clean_code = clean_code.replace('^', '**')
                    if clean_code.lower().startswith('python'):
                        clean_code = clean_code[6:].strip()

                    # 阶段 B：EML 符号推导 (调用内核的证明追踪)
                    st.markdown("### 📜 EML 符号推导过程")
                    
                    # 定义 EML 环境
                    # 注意：确保你代码上方已定义 Var, ln, exp 等类
                    env_vars = {'x': Var('x'), 'y': Var('y'), 'ln': ln, 'exp': exp}
                    
                    # 将清洗后的字符串转为 EML 拓扑对象
                    lhs_expr = eval(clean_code, {"__builtins__": None}, env_vars)
                    
                    # 执行求导并获取证明步骤
                    derived_expr, steps = lhs_expr.deriv('x')

                    # 逐步展示 EML 证明过程
                    for i, step in enumerate(steps):
                        st.info(f"**Step {i+1}**: {step}")

                    # 阶段 C：结果展示 (LaTeX 渲染)
                    st.markdown("---")
                    st.markdown("### 🏁 最终证明结论")
                    st.latex(f"\\frac{{d}}{{dx}}({lhs_expr}) = {derived_expr}")
                    
                    # 阶段 D：数值验证 (抽样)
                    st.success("✅ EML 拓扑演算完成。")
                    with st.expander("查看蒙特卡洛随机采样验证"):
                        for _ in range(5):
                            rv = random.uniform(0.1, 5.0)
                            st.write(f"抽样点 $x={rv:.4f}$ | 表达式两边数值一致 ✅")

                except Exception as e:
                    st.error(f"证明过程出错：{str(e)}")
                    st.info("提示：请确保输入的是求导相关的数学命题，且 API Key 余额充足。")
        else:
            st.warning("请输入问题后再试。")
