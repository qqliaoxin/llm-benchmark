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
├── run_benchmarks.py     # 自动化测试脚本，执行多轮压测
├── llm_benchmark.py      # 核心并发测试实现
├── README.md            # 项目文档
└── assets/              # 资源文件夹
```

## 组件说明

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

# 测试所有上下文大小（包括新增的大尺寸）

python3 context_benchmarks.py --llm_url http://localhost:11434/v1 --model deepseek-r1:32b --concurrency 1 --num_requests 1

python3 context_benchmarks.py --llm_url http://localhost:8000/v1 --model DeepSeek-R1 --concurrency 1 --num_requests 1 --context_sizes 13t --debug

# 单线程测试
python context_benchmarks.py --llm_url http://localhost:8000/v1 --concurrency 1

# 并发测试
python context_benchmarks.py --llm_url http://localhost:8000/v1 --concurrency 4

# 自定义参数
python context_benchmarks.py --llm_url http://localhost:8000/v1 --concurrency 2 --num_requests 5 --context_sizes 1k,4k,16k

# 测试所有上下文大小（包括13t）
python context_benchmarks.py --llm_url http://localhost:8000/v1 --model "DeepSeek-R1-0528-INT4"

# 仅测试13t
python context_benchmarks.py --llm_url http://localhost:8000/v1 --model "DeepSeek-R1-0528-INT4" --context_sizes 13t

# 测试13t和其他小尺寸
python context_benchmarks.py --llm_url http://localhost:8000/v1 --model "DeepSeek-R1-0528-INT4" --context_sizes "13t,1k,2k"

```bash
python run_benchmarks.py \
    --llm_url "http://192.168.0.136:8090/v1" \
    --api_key "your-api-key" \
    --model "DeepSeek-R1-0528-INT4" \
    --use_long_context
```

运行单次并发测试：

```bash
python llm_benchmark.py \
    --llm_url "http://192.168.0.136:8090/v1" \
    --api_key "your-api-key" \
    --model "DeepSeek-R1-0528-INT4" \
    --num_requests 100 \
    --concurrency 10
```

### 命令行参数

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
