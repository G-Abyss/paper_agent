#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证与清洗Agent
"""

from crewai import Agent, Task
from agents.base import get_llm


def create_abstract_validator_agent():
    """创建摘要验证 Agent（已弃用，保留用于兼容性）"""
    return create_abstract_validation_and_cleaning_agent()


def create_abstract_cleaner_agent():
    """创建摘要清洗 Agent（已弃用，保留用于兼容性）"""
    return create_abstract_validation_and_cleaning_agent()


def create_abstract_validation_and_cleaning_agent():
    """创建摘要验证与清洗 Agent（合并功能）"""
    return Agent(
        role="摘要验证与清洗专家",
        goal="验证摘要真实性并清洗格式，确保摘要内容真实可靠且格式规范",
        backstory="你是一位专业的学术内容验证与清洗专家。你具备双重能力：首先，你擅长识别AI生成的内容和真实提取的内容，能够通过分析文本特征、关键词、逻辑连贯性等来判断内容是否来自原文提取；其次，你擅长识别和清理学术文本中的无关内容，包括引用文献标记、图表引用、无意义的格式字符等。你能够在不改变原文意思的前提下，先验证内容真实性，然后清理干扰内容，使文本更加简洁易读。",
        allow_delegation=False,
        verbose=True,
        llm=get_llm(),
        max_iter=3,  # 合并任务可能需要更多迭代
        max_execution_time=300
    )


def create_abstract_validation_task(paper, abstract_text, source_type="网页"):
    """创建摘要验证任务（已弃用，保留用于兼容性）"""
    return create_abstract_validation_and_cleaning_task(paper, abstract_text, source_type)


def create_abstract_cleaning_task(abstract_text):
    """创建摘要清洗任务（已弃用，保留用于兼容性）"""
    # 这个函数需要paper对象，但为了兼容性，我们创建一个临时的paper对象
    paper = {'title': ''}
    return create_abstract_validation_and_cleaning_task(paper, abstract_text, "网页")


def create_abstract_validation_and_cleaning_task(paper, abstract_text, source_type="网页"):
    """创建摘要验证与清洗任务（合并功能）"""
    return Task(
        description=(
            f"## 任务目标\n"
            f"首先验证摘要内容的真实性，如果验证通过，则清洗摘要格式。这是一个两步任务：先验证，后清洗。\n\n"
            f"## 输入信息\n"
            f"**论文标题**：{paper['title']}\n\n"
            f"**提取的摘要内容**：\n```\n{abstract_text}\n```\n\n"
            f"**来源类型**：{source_type}\n\n"
            f"## 步骤1：验证摘要真实性\n"
            f"### 验证维度\n"
            f"#### 维度1：生成标志检测\n"
            f"检查摘要中是否包含明显的AI生成标志：\n"
            f"- 关键词标志：'simulated'、'generated'、'based on'、'not provided'、'assumed'、'presumed'、'inferred'等\n"
            f"- 元信息标志：'Note:'、'注意'、'说明'、'备注'等解释性文字\n"
            f"- 不确定性表述：'推测'、'假设'、'推断'、'可能'等不确定性的表述\n\n"
            f"#### 维度2：内容相关性验证\n"
            f"检查摘要内容与论文标题的相关性：\n"
            f"- 摘要是否明确提到标题中的关键概念和主题\n"
            f"- 摘要内容是否与标题主题高度一致\n"
            f"- 是否存在明显的主题偏离或无关内容\n\n"
            f"#### 维度3：文本特征分析\n"
            f"检查摘要是否具有真实学术摘要的特征：\n"
            f"- 是否包含具体的技术细节、方法、结果、贡献等实质性内容\n"
            f"- 是否过于笼统、模糊或缺乏具体信息\n"
            f"- 文本特征是否表明来自原文逐字提取（而非重新表述）\n\n"
            f"#### 维度4：逻辑连贯性评估\n"
            f"检查摘要的逻辑连贯性：\n"
            f"- 句子之间是否存在清晰的逻辑关系\n"
            f"- 是否包含完整的学术摘要结构（研究背景、方法、结果、结论等）\n"
            f"- 段落结构是否合理，信息流是否顺畅\n\n"
            f"### 验证判断标准\n"
            f"- **验证结果=1（真实可靠）**：摘要内容来自原文提取，通过所有验证维度，无明显AI生成标志。\n"
            f"- **验证结果=0（可能是虚构）**：摘要内容可能是AI虚构生成，或包含明显的生成标志，或未通过验证维度。\n\n"
            f"## 步骤2：清洗摘要格式（仅在验证结果=1时执行）\n"
            f"如果验证通过（验证结果=1），则对摘要进行格式清洗：\n\n"
            f"### 清洗规则\n"
            f"#### 规则1：删除引用文献标记\n"
            f"- 删除所有引用标记，包括但不限于：\n"
            f"  - 单引用：[1]、[2]、[3]等\n"
            f"  - 范围引用：[1-5]、[2-10]等\n"
            f"  - 多引用：[1,2,3]、[1, 2, 3]等\n"
            f"  - 混合引用：[1,3-5,7]等\n\n"
            f"#### 规则2：删除图表引用\n"
            f"- 删除所有图表引用，包括但不限于：\n"
            f"  - 中文格式：'图1'、'图2'、'表1'、'表2'等\n"
            f"  - 英文格式：'Figure 1'、'Fig. 1'、'Table 2'、'Tab. 2'等\n"
            f"  - 带括号格式：'(图1)'、'(Figure 1)'等\n\n"
            f"#### 规则3：清理无意义的换行\n"
            f"- 将多个连续的空行（≥2个）合并为单个空行\n"
            f"- 删除段落中间不必要的换行（保持段落内句子连续）\n"
            f"- 保留段落之间的合理分隔（单个空行）\n\n"
            f"#### 规则4：清理多余空格\n"
            f"- 删除行首和行尾的所有空格\n"
            f"- 将多个连续空格（≥2个）合并为单个空格\n"
            f"- 保留句子之间的单个空格\n\n"
            f"#### 规则5：保持内容完整性\n"
            f"- 仅删除标记和格式字符，禁止删除或修改实际的文本内容\n"
            f"- 禁止修改单词、句子或段落的实际内容\n"
            f"- 禁止添加、删除或重新组织文本内容\n\n"
            f"#### 规则6：保持逻辑连贯性\n"
            f"- 确保清洗后的文本逻辑连贯，句子完整\n"
            f"- 保持原文的段落结构和句子顺序\n"
            f"- 确保删除引用标记后，句子仍然语法正确\n\n"
            f"## 输出格式（严格遵循）\n"
            f"### 如果验证通过（验证结果=1）：\n"
            f"```\n"
            f"验证结果：1\n"
            f"验证说明：[简要说明验证依据，包括各维度的检查结果]\n"
            f"清洗后的摘要：\n[清洗后的摘要内容，已删除引用标记、图表引用和无意义的格式字符]\n"
            f"```\n\n"
            f"### 如果验证失败（验证结果=0）：\n"
            f"```\n"
            f"验证结果：0\n"
            f"验证说明：[详细说明检测到的虚构标志和未通过的验证维度]\n"
            f"清洗后的摘要：\n"
            f"```\n\n"
            f"## 注意事项\n"
            f"- 必须基于客观证据进行判断，避免主观臆测。\n"
            f"- 验证说明应具体、明确，指出具体的检测依据。\n"
            f"- 如果验证失败，不进行清洗，直接输出验证结果和说明。\n"
            f"- 如果验证通过，必须输出清洗后的摘要内容。\n"
            f"- 清洗后的摘要应保持原文的段落结构和句子顺序。"
        ),
        agent=create_abstract_validation_and_cleaning_agent(),
        expected_output="首先输出验证结果（1表示真实可靠，0表示可能是虚构的），然后输出验证说明，如果验证通过则输出清洗后的摘要内容"
    )

