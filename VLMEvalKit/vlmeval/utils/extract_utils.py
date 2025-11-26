import re

def extract_choice(ans: str) -> str:
    ans = ans.strip().upper()
    matches = re.findall(r"\b([A-I])\b", ans)
    if matches:
        return matches[-1]  # 返回最后一个合法选项字母
    return ""

def extract_answer(ans: str, max_option="E") -> str:
    if not ans:
        return ""
    ans = ans.strip()

    # 1. 先匹配合法选项字母
    matches = re.findall(rf"\b([A-{max_option}])\b", ans.upper())
    if matches:
        return matches[-1]  # 返回最后一个大写选项字母

    # 2. 否则提取一个单词或短语（允许字母+数字）
    phrase = re.findall(r"[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*", ans)
    if phrase:
        return phrase[0].strip()

    return "" 

def extract_yes_no(ans: str) -> str:
    ans = ans.strip().upper()
    matches = re.findall(r"\b(YES|NO)\b", ans)
    if matches:
        return matches[-1]  # 返回最后一个合法选项字母
    return "" 

def extract_first_word(ans: str) -> str:
    """
    提取输出中的第一行第一个单词
    """
    if not ans:
        return ""
    # 取第一行
    first_line = ans.strip().splitlines()[0]
    # 取第一个单词
    first_word = first_line.strip().split()[0]
    return first_word