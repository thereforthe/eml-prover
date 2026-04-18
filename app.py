import streamlit as st
import os
import random
import math
import re
import google.generativeai as genai

# ==========================================
# 1. EML 神经符号引擎 (内核：统一逻辑)
# ==========================================
class Expr:
    def __add__(self, other): return Add(self, _to_expr(other))
    def __mul__(self, other): return Mul(self, _to_expr(other))
    def __pow__(self, other): return Pow(self, _to_expr(other))
    def __truediv__(self, other): return Div(self, _to_expr(other))
    def __neg__(self): return Mul(Const(-1), self)  # 新增：支持负号 (-x)
    def __sub__(self, other): return Add(self, -_to_expr(other))  # 新增：支持减法 (a - b)
    def deriv(self, var): raise NotImplementedError

def _to_expr(val):
    if isinstance(val, Expr): return val
    return Const(val)

class Const(Expr):
    def __init__(self, val):
        [span_5](start_span)
        self.val = val  # 确保这一行是完整的[span_5](end_span)
    def deriv(self, var):
        [span_6](start_span)
        return Const(0), [f"根据常数规则: $d({self.val})/d{var} = 0$"][span_6](end_span)

class Var(Expr):
    def __init__(self, name): self.name = name
    def deriv(self, var):
        if self.name == var:
            return Const(1), [f"基础公理: $d({self.name})/d{self.name} = 1$"]
        return Const(0), [f"基础公理: $d({self.name})/d{var} = 0$"]
    def eval(self, env): return env.get(self.name, 0)
    def __str__(self): return self.name

class Add(Expr):
    def __init__(self, left, right): self.left, self.right = left, right

    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        [span_10](start_span);
        return ld + rd, ls + rs + [f"线性规则: $d({self.left} + {self.right}) = d({self.left}) + d({self.right})$"][
            span_10](end_span)

class Mul(Expr):
    def __init__(self, left, right): self.left, self.right = left, right

    def deriv(self, var):
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)
        res = ld * self.right + self.left * rd
        [span_12](start_span);
        return res, ls + rs + [
            f"乘法法则(Leibniz): $d({self.left} \\cdot {self.right}) = ({ld})({self.right}) + ({self.left})({rd})$"][
            span_12](end_span)

class Pow(Expr):
    def __init__(self, left, right):
        self.left, self.right = left, right
    def deriv(self, var):
        if isinstance(self.right, Const):
            n = self.right.val
            ld, ls = self.left.deriv(var)
            res = Const(n) * (self.left ** Const(n-1)) * ld
            return res, ls + [f"幂链式法则: $d(u^{{{n}}}) = {n}u^{{{n-1}}} \\cdot du$"]
        return Const(0), ["目前仅支持常数幂求导"]
    def eval(self, env):
        return self.left.eval(env) ** self.right.eval(env)
    def __str__(self):
        return f"({self.left})^{{{self.right}}}"

class Exp(Expr):
    def __init__(self, inner): self.inner = inner

    def deriv(self, var):
        id, is_ = self.inner.deriv(var)
        res = exp(self.inner) * id
        [span_14](start_span);
        return res, is_ + [f"指数规则: $d(e^{{{self.inner}}}) = e^{{{self.inner}}} \\cdot d({self.inner})$"][span_14](
            end_span)

def exp(x): return Exp(_to_expr(x))

class Ln(Expr):
    def __init__(self, inner): self.inner = inner

    def deriv(self, var):
        id, is_ = self.inner.deriv(var)
        res = id / self.inner
        [span_16](start_span);
        return res, is_ + [
            f"对数链式法则: $d(\\ln({self.inner})) = \\frac{{1}}{{{self.inner}}} \\cdot d({self.inner}) = {res}$"][
            span_16](end_span)


class Div(Expr):
    def __init__(self, left, right):
        self.left, self.right = left, right

    def deriv(self, var):
        # 1. 递归求子项导数
        ld, ls = self.left.deriv(var)
        rd, rs = self.right.deriv(var)

        # 2. 直接使用数学符号计算结果对象 (得益于 Expr 里的 __sub__)
        res = (ld * self.right - self.left * rd) / (self.right * self.right)

        # 3. 返回结果和步骤列表
        return res, ls + rs + [f"商法则: $d({self.left}/{self.right}) = \\frac{{u'v - uv'}}{{v^2}}$"]

    def eval(self, env):
        return self.left.eval(env) / self.right.eval(env)

    def __str__(self):
        return f"\\frac{{{self.left}}}{{{self.right}}}"

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

    user_input = st.text_input("输入你想证明的问题：", placeholder="例如：证明 ln(x^2) 的导数是 2/x")

    if st.button("执行 EML 符号证明"):
        if user_input:
            with st.spinner("AI 正在解析语义并构建 EML 拓扑树..."):
                try:
                    # 阶段 A：语义编译
                    prompt = (
                        "Translate the math problem into a Python expression for the LHS. "
                        "Variables: x, y. Functions: ln, exp. Output ONLY the expression, "
                        f"no backticks, no Chinese punctuation. Problem: {user_input}"
                    )
                    raw_text = model.generate_content(prompt).text.strip()

                    # 强力清洗（修复：正则表达式语法错误）
                    clean_code = re.sub(r'span_\d+', '', raw_text)
                    clean_code = re.sub(r"'", '', clean_code)  # 修复：正确去除单引号的写法
                    clean_code = clean_code.replace('```python', '').replace('```', '').replace('`', '').strip()
                    clean_code = clean_code.replace('，', ',').replace('（', '(').replace('）', ')')
                    clean_code = clean_code.replace('^', '**')
                    if clean_code.lower().startswith('python'): clean_code = clean_code[6:].strip()

                    # 阶段 B：EML 符号推导过程
                    st.markdown("### 📜 EML 符号推导过程")
                    env_vars = {'x': Var('x'), 'y': Var('y'), 'ln': ln, 'exp': exp}

                    # 执行符号转换
                    lhs_expr = eval(clean_code, {"__builtins__": None}, env_vars)

                    # 执行求导并获取证明步骤
                    derived_expr, steps = lhs_expr.deriv('x')

                    # 展示每一个步骤
                    for i, step in enumerate(steps):
                        st.info(f"**Step {i + 1}**: {step}")

                    # 阶段 C：结果展示
                    st.markdown("---")
                    st.markdown("### 🏁 最终证明结论")
                    st.latex(f"\\frac{{d}}{{dx}}({lhs_expr}) = {derived_expr}")
                    st.success("✅ EML 推导完成")

                except Exception as e:
                    st.error(f"证明出错：{str(e)}")
        else:
            st.warning("请输入问题后再试。")
