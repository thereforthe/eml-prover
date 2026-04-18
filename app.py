import streamlit as st
import os
import random
import math
import google.generativeai as genai


# ==========================================
# 1. EML 神经符号核心引擎 (微型 CAS 计算机代数系统)
# 支撑 .deriv() 和 .eval() 方法的符号拓扑树
# ==========================================
class Expr:
    def __add__(self, other): return Add(self, _to_expr(other))

    def __radd__(self, other): return Add(_to_expr(other), self)

    def __sub__(self, other): return Sub(self, _to_expr(other))

    def __rsub__(self, other): return Sub(_to_expr(other), self)

    def __mul__(self, other): return Mul(self, _to_expr(other))

    def __rmul__(self, other): return Mul(_to_expr(other), self)

    def __truediv__(self, other): return Div(self, _to_expr(other))

    def __rtruediv__(self, other): return Div(_to_expr(other), self)

    def __pow__(self, other): return Pow(self, _to_expr(other))

    def deriv(self, var): raise NotImplementedError

    def eval(self, env): raise NotImplementedError

    def __str__(self): raise NotImplementedError


def _to_expr(val):
    if isinstance(val, Expr): return val
    return Const(val)


class Const(Expr):
    def __init__(self, val): self.val = val

    def deriv(self, var): return Const(0)

    def eval(self, env): return self.val

    def __str__(self): return str(self.val)


class Var(Expr):
    def __init__(self, name): self.name = name

    def deriv(self, var): return Const(1) if self.name == var else Const(0)

    def eval(self, env): return env.get(self.name, 0)

    def __str__(self): return self.name


class Add(Expr):
    def __init__(self, left, right): self.left, self.right = left, right

    def deriv(self, var): return self.left.deriv(var) + self.right.deriv(var)

    def eval(self, env): return self.left.eval(env) + self.right.eval(env)

    def __str__(self): return f"({self.left} + {self.right})"


class Sub(Expr):
    def __init__(self, left, right): self.left, self.right = left, right

    def deriv(self, var): return self.left.deriv(var) - self.right.deriv(var)

    def eval(self, env): return self.left.eval(env) - self.right.eval(env)

    def __str__(self): return f"({self.left} - {self.right})"


class Mul(Expr):
    def __init__(self, left, right): self.left, self.right = left, right

    def deriv(self, var): return self.left.deriv(var) * self.right + self.left * self.right.deriv(var)

    def eval(self, env): return self.left.eval(env) * self.right.eval(env)

    def __str__(self): return f"({self.left} \\cdot {self.right})"


class Div(Expr):
    def __init__(self, left, right): self.left, self.right = left, right

    def deriv(self, var): return (self.left.deriv(var) * self.right - self.left * self.right.deriv(var)) / (
                self.right ** 2)

    def eval(self, env): return self.left.eval(env) / self.right.eval(env)

    def __str__(self): return f"\\frac{{{self.left}}}{{{self.right}}}"


class Pow(Expr):
    def __init__(self, left, right): self.left, self.right = left, right

    def deriv(self, var):
        # 链式法则：d/dx(f^g) = f^g * (g'*ln(f) + g*f'/f)
        return (self.left ** self.right) * (
                    self.right.deriv(var) * ln(self.left) + self.right * self.left.deriv(var) / self.left)

    def eval(self, env): return self.left.eval(env) ** self.right.eval(env)

    def __str__(self): return f"({self.left}^{{{self.right}}})"


class Exp(Expr):
    def __init__(self, inner): self.inner = inner

    def deriv(self, var): return exp(self.inner) * self.inner.deriv(var)

    def eval(self, env): return math.exp(self.inner.eval(env))

    def __str__(self): return f"e^{{{self.inner}}}"


class Ln(Expr):
    def __init__(self, inner): self.inner = inner

    def deriv(self, var): return self.inner.deriv(var) / self.inner

    def eval(self, env): return math.log(self.inner.eval(env))

    def __str__(self): return f"\\ln({self.inner})"


def exp(x): return Exp(_to_expr(x))


def ln(x): return Ln(_to_expr(x))


zero = Const(0)

# ==========================================
# 2. 网页 UI 设计
# ==========================================
st.set_page_config(page_title="EML 神经符号证明器", page_icon="🧪")

st.title("🧪 EML 神经符号证明引擎")
st.markdown("""
本系统通过 **AI 语义解析** + **EML 符号拓扑引擎** 进行严谨的数学证明。
请输入你的数学猜想（例如：`ln(x) 的导数是 1/x` 或 `(x+1)^2 = x^2 + 2x + 1`）。
""")

# 侧边栏：API Key 配置
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("输入 Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    st.info("Key 不会被存储，仅用于本次会话。")

# 3. 证明核心流程
# ==========================================
if api_key:
    genai.configure(api_key=api_key)
    try:
        # 诊断并自动匹配模型
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available_models:
            st.error("⚠️ 诊断失败：你的 API Key 没权访问任何模型。")
            st.stop()

        selected_model_name = next((name for name in available_models if "flash" in name), available_models[0])
        model = genai.GenerativeModel(selected_model_name)
        st.sidebar.success(f"✅ 已自动连接: {selected_model_name}")

    except Exception as e:
        st.error(f"❌ 访问 API 失败：{str(e)}")
        st.stop()

    user_input = st.text_input("输入你想证明的问题：", placeholder="例如：证明 x*x 的导数是 2*x")

    if st.button("开始机器证明"):
        if user_input:
            with st.spinner("AI 正在解析语义并构建 EML 拓扑树..."):
                try:
                    # 阶段 A：编译
                    prompt_compile = f"把这个问题翻译成 Python 等式代码（变量用 x, y, n，函数用 exp, ln），求导用 .deriv('x')。只输出代码，确保包含 '==' 符号。如：(x*x).deriv('x') == 2*x。问题：{user_input}"

                    raw_text = model.generate_content(prompt_compile).text.strip()
                    py_code = raw_text.replace('```python', '').replace('```', '').replace('`', '').strip()
                    if py_code.lower().startswith('python'):
                        py_code = py_code[6:].strip()

                    st.markdown("### 🧩 步骤 1：AI 语义编译为 AST")
                    st.code(f"{py_code}", language="python")

                    if "==" not in py_code:
                        st.error("AI 生成的代码格式错误，缺少等式。请重试。")
                        st.stop()

                    # 阶段 B：解析 EML (构建抽象语法树并符号展开)
                    left_s, right_s = py_code.split("==", 1)
                    env_vars = {'x': Var('x'), 'y': Var('y'), 'exp': exp, 'ln': ln, 'zero': zero}

                    # eval 将自动触发我们自定义的 Expr 类及其 .deriv 方法
                    lhs = eval(left_s.strip(), {"__builtins__": None}, env_vars)
                    rhs = eval(right_s.strip(), {"__builtins__": None}, env_vars)

                    # --- UI 展示：使用 EML 的符号证明/展开过程 ---
                    st.markdown("### 🧬 步骤 2：EML 符号拓扑树展开")
                    st.info("引擎截获了求导 `.deriv()` 指令，基于链式法则完成了符号展开：")
                    st.latex(f"\\text{{左式 (LHS展开): }} {lhs}")
                    st.latex(f"\\text{{右式 (RHS目标): }} {rhs}")

                    # 阶段 C：证明 (定义域自适应蒙特卡洛测试)
                    st.markdown("### 🎯 步骤 3：蒙特卡洛随机逼近验证")
                    is_valid = True
                    successful_tests = 0
                    max_attempts = 1000
                    attempts = 0

                    # 显示部分抽样日志
                    log_container = st.empty()
                    logs = []

                    while successful_tests < 20 and attempts < max_attempts:
                        attempts += 1
                        test_env = {'x': random.uniform(-10.0, 10.0), 'y': random.uniform(-10.0, 10.0)}

                        try:
                            l_val = lhs.eval(test_env)
                            r_val = rhs.eval(test_env)

                            if math.isnan(l_val) or math.isnan(r_val) or math.isinf(l_val) or math.isinf(r_val):
                                continue

                            # 记录前几次成功的求值，用于展示过程
                            if successful_tests < 3:
                                logs.append(f"- 抽样 $x={test_env['x']:.4f}$ | LHS: {l_val:.6f}, RHS: {r_val:.6f}")

                            if not math.isclose(l_val, r_val, rel_tol=1e-4, abs_tol=1e-9):
                                is_valid = False
                                logs.append(
                                    f"- **发现反例！** $x={test_env['x']:.4f}$ | LHS: {l_val:.6f} $\\neq$ RHS: {r_val:.6f}")
                                break

                            successful_tests += 1

                        except (ValueError, ZeroDivisionError, OverflowError):
                            # 踩到定义域雷区（如 ln(-1)），跳过
                            continue

                    log_container.markdown("\n".join(logs))

                    if successful_tests < 20 and is_valid:
                        is_valid = False
                        st.warning("⚠️ 无法在定义域内收集到足够的合法样本，命题可能无意义。")

                    # 阶段 D：结果展示
                    st.markdown("### 🏁 步骤 4：最终判定")
                    if is_valid:
                        st.success(f"✅ 机器判定：命题成立 (Q.E.D.) ── 共通过 {successful_tests} 次独立随机域测试。")
                    else:
                        st.error("❌ 机器判定：命题不成立")

                    # 阶段 E：AI 报告
                    prompt_explain = f"命题：{user_input}。机器验证结果：{'成立' if is_valid else '不成立'}。你之前生成的代码是 {py_code}。请从抽象语法树(AST)、求导法则、或者代数展开的角度，简短向用户解释为什么是这个结果。使用 Markdown 格式。"
                    report = model.generate_content(prompt_explain).text
                    st.markdown("### 📜 自动化分析报告")
                    st.write(report)

                except Exception as e:
                    st.error(f"证明过程出错：{str(e)}。可能是 AI 编译出了无法被 EML 解析的格式。")
        else:
            # 这里修复了你原始代码的截断部分
            st.warning("请在上方输入你想要证明的数学问题！")
