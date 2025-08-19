# LLM-Benchmark

LLM 并发性能测试工具，支持自动化压力测试和性能报告生成。

## 功能特点

- 多阶段并发测试（从低并发逐步提升到高并发）
- 自动化测试数据收集和分析
- 详细的性能指标统计和可视化报告
- 支持短文本和长文本测试场景
- 灵活的配置选项
- 生成 JSON 输出以便进一步分析或可视化

## 项目结构

```
llm-benchmark/
├── context_benchmarks.py # 上下文性能测试工具
├── run_benchmarks.py     # 自动化压力测试脚本
├── llm_benchmark.py      # 核心并发测试实现
├── README.md            # 项目文档
└── assets/              # 资源文件夹
```

## 组件说明

- **context_benchmarks.py**:
  - 测试不同上下文大小下的模型性能
  - 支持多种上下文规模（13t到128k）
  - 提供详细的性能指标分析
  - 生成美观的测试报告

- **run_benchmarks.py**:
  - 执行多轮自动化压力测试
  - 自动调整并发配置（1-300 并发）
  - 收集和汇总测试数据
  - 生成美观的性能报告

- **llm_benchmark.py**:
  - 实现核心并发测试逻辑
  - 管理并发请求和连接池
  - 收集详细性能指标
  - 支持流式响应测试

## 使用方法

### 1. 上下文性能测试 (context_benchmarks.py)

测试模型在不同上下文大小下的性能表现：

```bash
# 基础测试
python context_benchmarks.py --llm_url http://localhost:8000/v1 --model "DeepSeek-R1"

# 自定义上下文大小
python context_benchmarks.py --llm_url http://localhost:8000/v1 --model "DeepSeek-R1" --context_sizes "1k,4k,16k"

# 并发测试
python context_benchmarks.py --llm_url http://localhost:8000/v1 --model "DeepSeek-R1" --concurrency 4

# 调试模式
python context_benchmarks.py --llm_url http://localhost:8000/v1 --model "DeepSeek-R1" --debug
```

### 2. 自动化压力测试 (run_benchmarks.py)

执行多轮自动化压力测试：

```bash
python run_benchmarks.py \
    --llm_url "http://localhost:8000/v1" \
    --api_key "your-api-key" \
    --model "DeepSeek-R1" \
    --use_long_context
```

### 3. 单次并发测试 (llm_benchmark.py)

运行单次并发性能测试：

```bash
python llm_benchmark.py \
    --llm_url "http://localhost:8000/v1" \
    --api_key "your-api-key" \
    --model "DeepSeek-R1" \
    --num_requests 100 \
    --concurrency 10
```

### 命令行参数

#### context_benchmarks.py 参数

| 参数               | 说明                           | 默认值                                    |
| ------------------ | ------------------------------ | ----------------------------------------- |
| --llm_url          | LLM 服务器 URL                 | 必填                                      |
| --api_key          | API 密钥                       | default                                   |
| --model            | 模型名称                       | deepseek-r1                               |
| --context_sizes    | 测试的上下文大小（逗号分隔）   | 13t,1k,2k,4k,8k,16k,32k,64k,92k,128k      |
| --num_requests     | 每个上下文大小的请求次数       | 3                                         |
| --output_tokens    | 输出 token 数量                | 200                                       |
| --request_timeout  | 请求超时时间（秒）             | 120                                       |
| --concurrency      | 并发请求数                     | 1                                         |
| --debug            | 启用调试模式                   | False                                     |
| --skip_sse_test    | 跳过 SSE 连接测试              | False                                     |

#### run_benchmarks.py 参数

| 参数               | 说明               | 默认值      |
| ------------------ | ------------------ | ----------- |
| --llm_url          | LLM 服务器 URL     | 必填        |
| --api_key          | API 密钥           | 选填        |
| --model            | 模型名称           | deepseek-r1 |
| --use_long_context | 使用长文本测试模式 | False       |

#### llm_benchmark.py 参数

| 参数              | 说明                | 默认值      |
| ----------------- | ------------------- | ----------- |
| --llm_url         | LLM 服务器 URL      | 必填        |
| --api_key         | API 密钥            | 选填        |
| --model           | 模型名称            | deepseek-r1 |
| --num_requests    | 总请求数            | 必填        |
| --concurrency     | 并发数              | 必填        |
| --output_tokens   | 输出 token 数限制   | 50          |
| --request_timeout | 请求超时时间(秒)    | 60          |
| --output_format   | 输出格式(json/line) | line        |

## 测试报告示例

![性能测试报告示例](./assets/image-20250220155605371.png)

## 开源许可

本项目采用 [MIT License](LICENSE) 开源协议。
