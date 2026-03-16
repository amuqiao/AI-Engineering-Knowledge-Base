我把之前所有Ray核心知识、速查、实战模板、生产规范整合为一篇**结构化、可直接留存复用**的完整笔记，按入门→核心→实战→生产的逻辑梳理，方便随时查阅和落地。
# Ray 分布式框架全栈笔记（入门+实战+生产）

# 一、Ray 框架基础认知

## 1.1 核心定义

**Ray**是面向Python的**高性能分布式计算框架**，专为AI大模型、并行计算、强化学习、批量数据处理设计，解决Python GIL锁无法多核并行、分布式代码编写复杂的痛点，极简API即可实现单机多核/多机集群并行。

## 1.2 核心优势

- **极简接入**：仅需装饰器，无需改动业务逻辑

- **高性能**：低延迟、高吞吐，任务调度极致高效

- **资源灵活调度**：CPU/GPU/内存精细化管控

- **场景全覆盖**：大模型推理、训练、数据处理、服务部署

- **跨环境兼容**：本地/集群/云服务代码无缝迁移

## 1.3 核心术语速记

- **远程任务**：@ray.remote修饰的函数，分布式异步执行

- **Actor**：@ray.remote修饰的类，带状态的分布式服务，适合模型复用

- **Future**：任务提交后返回的句柄，异步占位

- **ray.get()**：阻塞获取任务结果

- **ray.wait()**：批量等待任务，支持超时控制

---

# 二、Ray 基础速查手册（高频指令）

## 2.1 安装命令

```bash
pip install ray  # 基础安装
pip install ray[default]  # 完整功能（含监控、集群）
```

## 2.2 初始化与关闭

```python
import ray

# 本地默认启动（自动识别核心数）
ray.init()

# 自定义资源/日志级别
ray.init(num_cpus=4, num_gpus=1, logging_level="ERROR")

# 本地调试模式（不并行，方便排查bug）
ray.init(local_mode=True)

# 连接多机集群
ray.init(address="ray://主节点IP:10001")

# 关闭Ray
ray.shutdown()
```

## 2.3 分布式函数（无状态任务）

```python
# 定义远程任务
@ray.remote
def calc(a, b):
    return a + b

# 提交任务（异步，不阻塞）
future = calc.remote(1, 2)

# 获取结果（阻塞等待）
res = ray.get(future)

# 批量并行任务
futures = [calc.remote(i, i+1) for i in range(10)]
results = ray.get(futures)
```

## 2.4 分布式Actor（有状态服务）

```python
# 适合模型加载、状态缓存，全局仅初始化一次
@ray.remote(num_gpus=1)
class ModelActor:
    def __init__(self):
        self.model = self.load_model()  # 模型仅加载1次，节省显存

    def load_model(self):
        # 替换为实际模型加载逻辑
        return "loaded_model"

    def infer(self, data):
        return f"推理结果：{data}"

# 创建Actor实例
actor = ModelActor.remote()
# 调用Actor方法
result = ray.get(actor.infer.remote("测试数据"))
```

## 2.5 任务管控与资源指定

```python
# 指定CPU/GPU资源
@ray.remote(num_cpus=2, num_gpus=1)

# 批量任务超时控制
done, pending = ray.wait(futures, num_returns=2, timeout=5)

# 取消任务
ray.cancel(future)

# 查看集群资源
ray.cluster_resources()  # 总资源
ray.available_resources()  # 可用资源
```

---

# 三、Ray 实战模板（开箱即用）

所有模板可直接复制运行，适配本地/集群，生产环境可直接复用

## 3.1 基础并行任务模板（耗时任务加速）

```python
import ray
import time

# 初始化
ray.init(num_cpus=4)

# 模拟耗时任务
@ray.remote
def heavy_task(n):
    time.sleep(1)
    return f"任务{n}完成"

if __name__ == "__main__":
    # 并行提交5个任务，串行需5s，并行仅需1s
    futures = [heavy_task.remote(i) for i in range(5)]
    results = ray.get(futures)
    print(results)
```

## 3.2 大模型分布式推理模板（核心场景）

```python
import ray
from transformers import AutoTokenizer, AutoModelForCausalLM

# 初始化（无GPU去掉num_gpus参数）
ray.init(num_gpus=1)

# 模型Actor：全局加载一次，避免重复显存占用
@ray.remote(num_gpus=1)
class LLMInfer:
    def __init__(self):
        self.model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs, max_new_tokens=64, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model = LLMInfer.remote()
    prompts = ["AI is going to", "Ray framework is best for"]
    tasks = [model.generate.remote(p) for p in prompts]
    results = ray.get(tasks)
    for p, r in zip(prompts, results):
        print(f"\n输入：{p}\n输出：{r}")
```

## 3.3 Ray + FastAPI 大模型API服务模板

```python
from fastapi import FastAPI
import ray
from pydantic import BaseModel

# 初始化Ray，忽略重复初始化错误
ray.init(ignore_reinit_error=True)

# 模型Actor
@ray.remote(num_gpus=1)
class LLMModel:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model="distilgpt2")

    def generate(self, prompt):
        return self.pipe(prompt, max_new_tokens=50)[0]["generated_text"]

# 全局加载模型
model = LLMModel.remote()

# FastAPI服务
app = FastAPI(title="Ray LLM 分布式推理服务")

class InferRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
async def run_infer(req: InferRequest):
    result = ray.get(model.generate.remote(req.prompt))
    return {"prompt": req.prompt, "response": result}

# 启动命令：uvicorn main:app --host 0.0.0.0 --port 8000
```

## 3.4 Ray + vLLM 高吞吐推理模板

```python
import ray
from vllm import LLM, SamplingParams

# 安装vllm：pip install vllm
ray.init(num_gpus=1)

@ray.remote(num_gpus=1)
class VLLMService:
    def __init__(self):
        self.llm = LLM(model="distilgpt2", gpu_memory_utilization=0.8)
        self.sampling = SamplingParams(max_tokens=64)

    def batch_infer(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling)
        return [out.outputs[0].text for out in outputs]

if __name__ == "__main__":
    service = VLLMService.remote()
    prompts = ["AI will", "Ray is", "Code generation"]
    results = ray.get(service.batch_infer.remote(prompts))
    for p, r in zip(prompts, results):
        print(f"输入：{p} | 输出：{r}")
```

## 3.5 多机集群部署模板

### 3.5.1 主节点（Head Node）启动

```bash
ray start --head --port=6379 --dashboard-host 0.0.0.0
```

### 3.5.2 子节点（Worker Node）加入集群

```bash
ray start --address=主节点IP:6379
```

### 3.5.3 代码连接集群

```python
import ray
# 连接集群，后续代码与本地完全一致
ray.init(address="ray://主节点IP:10001")

# 测试集群任务
@ray.remote
def test_cluster():
    return "集群任务执行成功"

print(ray.get(test_cluster.remote()))
```

## 3.6 生产者-消费者任务调度模板

```python
import ray
from ray.util.queue import Queue

ray.init()

# 生产者：投放任务
@ray.remote
def producer(queue):
    for i in range(10):
        queue.put(f"task-{i}")
    queue.put(None)  # 结束信号

# 消费者：处理任务
@ray.remote
def consumer(queue):
    res = []
    while True:
        task = queue.get()
        if task is None:
            queue.put(None)
            break
        res.append(task)
    return res

if __name__ == "__main__":
    q = Queue()
    ray.get(producer.remote(q))
    results = ray.get(consumer.remote(q))
    print("任务结果：", results)
```

---

# 四、Ray 生产环境最佳实践

## 4.1 资源管控规范

- 模型加载必须放在**Actor**中，禁止在远程函数内加载，避免显存/内存重复浪费

- 精细化分配资源：@ray.remote(num_cpus=2, num_gpus=0.5)（GPU共享）

- 单机多任务避免资源抢占，合理设置CPU/GPU配额

## 4.2 稳定性保障

- 批量任务必须加**超时控制**，防止任务阻塞

- 使用ray.wait()分批处理任务，避免内存溢出

- 服务部署前用local_mode=True调试，排查逻辑bug

## 4.3 监控与运维

```bash
# 启动Ray监控面板
ray dashboard --host 0.0.0.0  # 访问：http://IP:8265

# 查看集群状态
ray status
```

## 4.4 避坑指南

- 禁止在循环中重复创建Actor，Actor全局单例复用

- 大模型推理优先用vLLM+Ray，提升10-20倍吞吐

- 集群部署确保所有节点Ray版本一致

- 无GPU环境去掉num_gpus参数，避免启动失败

---

# 五、核心口诀（快速记忆）

**函数加@ray.remote，调用用.remote()，结果用ray.get()**

**模型放进Actor里，全局复用省显存**

**本地集群一套码，并行加速全靠它**

需要我帮你**标注重点高亮句**，让笔记更易快速抓核心吗？