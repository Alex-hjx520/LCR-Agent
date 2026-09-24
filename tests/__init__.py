"""测试包。

约定：
- 单元测试不依赖网络、不下载模型、不调用真实 LLM；
- 需要真实依赖（向量模型 / OpenAI）的用例统一打 `@pytest.mark.integration`。
"""
