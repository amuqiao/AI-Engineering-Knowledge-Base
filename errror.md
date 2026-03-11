
根据代码分析，我发现了导致服务器重启的关键问题。让我创建一个详细的分析报告：

## 问题分析报告

通过分析代码，我发现了**3 个关键问题**导致服务器在处理大量任务时重启：

### 🔴 **问题 1：数据库连接未正确释放（最严重）**

在 [`ocr_qr_task_router.py`](file:///data/home/project/servo_ai_project/servo_ai/src/routers/ocr/ocr_qr_task_router.py#L413-L444) 的 `batch_refresh_status` 函数中：

```python
async def batch_refresh_status(...):
    try:
        records = await OCRService.fetch_ocr_records_with_conditions(db, ...)
        for record in records:
            new_ai_status = _check_result(record.ai_content)
            OCRService.update_ai_status(record.id, new_ai_status, db)  # 每次都 commit
            record_ids.append(record.id)
```

**问题：**
- 每个记录更新后都调用 `db.commit()`，导致**事务无法批量提交**
- 当处理 7000 条记录时，会产生 7000 次数据库提交操作
- 数据库连接长时间占用，连接池可能耗尽

### 🔴 **问题 2：内存泄漏 - 图片处理未释放**

在 [`ocr_cert_qr_task_v3.py`](file:///data/home/project/servo_ai_project/servo_ai/src/services/tasks/ocr_cert_qr_task_v3.py#L491-L566) 的 `_try_multiple_methods` 方法中：

```python
def _try_multiple_methods(self, img: Image.Image):
    # 方法 2：图像缩小、放大
    scales = [0.5, 2.0, 3.0]
    for scale in scales:
        scaled_img = img.resize(new_size, Image.Resampling.LANCZOS)
        scaled_objects = qr_decode(np.array(scaled_img))
        # 虽然有 finally del，但 np.array() 创建的数组没有显式释放
```

**问题：**
- `np.array(scaled_img)` 创建的 numpy 数组没有显式删除
- 处理 7000 个任务，每个任务多个 URL，每个 URL 多张图片，会产生大量内存碎片
- 虽然有 `del scaled_img`，但 numpy 数组的内存没有释放

### 🔴 **问题 3：批量任务创建时数据库查询次数过多**

在 [`ocr_qr_task_router.py`](file:///data/home/project/servo_ai_project/servo_ai/src/routers/ocr/ocr_qr_task_router.py#L359-L400) 的 `create_ocr_tasks_refresh_status` 接口中：

```python
batch_size = 100  # 批次太小
handled_num = 0
while len(batch_record_ids) == batch_size:
    # 循环查询数据库
    batch_record_ids = await batch_refresh_status(...)
```

**问题：**
- `batch_size = 100` 太小，处理 7000 条记录需要循环 70 次
- 每次循环都查询数据库，造成大量数据库 IO
- 对比 `refresh-data` 接口的 `batch_size = 500`，这个更小

### 🔴 **问题 4：数据库连接生成器未正确关闭**

在 [`ocr_cert_qr_task_v3.py`](file:///data/home/project/servo_ai_project/servo_ai/src/services/tasks/ocr_cert_qr_task_v3.py#L94-L125) 的 `process` 方法中：

```python
def process(self) -> Dict[str, Any]:
    db = next(get_db_conn())
    try:
        # ... 处理逻辑
    finally:
        db.close()  # 只关闭了 db，但没有关闭生成器
```

**问题：**
- `get_db_conn()` 是一个生成器，需要使用 `try...finally` 确保生成器完整执行
- 当前代码只调用了 `db.close()`，但生成器没有执行到 `finally` 块
- 导致数据库连接没有正确归还到连接池

### 📊 **对比：为什么 3-4 万个备案证不会重启**

之前的接口（如 `/tasks/by-business-ids`）：
1. 直接查询后发布任务到 Redis，**不立即执行**
2. 任务由 Celery Worker 异步处理，**内存压力分散**
3. 数据库连接使用后立即释放

现在的 `/tasks/refresh-status` 接口：
1. **同步执行**所有 7000 条记录的状态检查和更新
2. 每条记录都进行 JSON 解析和字段检查
3. 每条记录都更新数据库并提交
4. **所有操作在一个 HTTP 请求周期内完成**，内存无法释放

### 💡 **解决方案**

#### ✅ **方案 1：优化数据库连接管理（最高优先级）**

**问题根源：** 在 `ocr_cert_qr_task.py` 的 `process` 方法中，数据库连接生成器没有正确关闭

**修复代码：**

```python
# src/services/tasks/ocr_cert_qr_task.py 的 process 方法
def process(self) -> Dict[str, Any]:
    """任务处理逻辑"""
    # 手动获取数据库会话
    db_generator = get_db_conn()
    db = next(db_generator)
    try:
        logger.info(f"开始处理识别二维码任务，ID: {self.task_id}，内容：{self.content}")
        # ... 原有处理逻辑 ...
        
        return {
            "status": "success",
            "task_id": self.task_id,
            "result": processed_result
        }
    finally:
        # ✅ 正确关闭数据库连接和生成器
        try:
            db.close()
        except Exception as e:
            logger.error(f"关闭数据库连接失败：{str(e)}")
        try:
            next(db_generator)  # 推进生成器到 finally 块，确保连接归还
        except StopIteration:
            pass  # 生成器已正常结束
        del db, db_generator  # 显式清理变量
```

**修复代码：** `ocr_tasks.py` 中的 `fetch_and_publish_latest_ocr_tasks` 任务也需要同样修复

```python
# src/celery_app/ocr_tasks.py
@shared_task(name='celery_app.tasks.fetch_and_publish_latest_ocr_tasks', bind=True, max_retries=3)
def fetch_and_publish_latest_ocr_tasks(self, limit_count: int = 100):
    """获取最新 OCR 记录并发布处理任务"""
    try:
        # ... 原有代码 ...
        return True
    except Exception as e:
        logger.error(f"获取并发布 OCR 任务失败：{str(e)}", exc_info=True)
        self.retry(exc=e)
    finally:
        # ✅ 确保数据库连接生成器完成以释放连接
        if db_generator:
            try:
                next(db_generator)
            except StopIteration:
                pass
        # 确保 Redis 连接生成器完成以释放连接
        if redis_generator:
            try:
                next(redis_generator)
            except StopIteration:
                pass
        # 显式清理变量
        del db, redis_client, db_generator, redis_generator
```

---

#### ✅ **方案 2：优化内存管理 - 显式释放 numpy 数组**

**问题根源：** 在 `_try_multiple_methods` 方法中，numpy 数组没有显式释放

**修复代码：**

```python
# src/services/tasks/ocr_cert_qr_task_v3.py 的 _try_multiple_methods 方法
def _try_multiple_methods(self, img: Image.Image) -> List[Any]:
    """尝试多种方法识别二维码"""
    # 方法 1：直接原图识别
    img_array = np.array(img)
    try:
        decoded_objects = qr_decode(img_array)
        if decoded_objects:
            logger.info(f"原图识别成功，找到{len(decoded_objects)}个二维码")
            return decoded_objects
    finally:
        del img_array  # ✅ 显式删除 numpy 数组
    
    # 方法 2：图像缩小、放大
    scales = [0.5, 2.0, 3.0]
    for scale in scales:
        try:
            new_size = (int(img.width * scale), int(img.height * scale))
            scaled_img = img.resize(new_size, Image.Resampling.LANCZOS)
            scaled_array = np.array(scaled_img)
            try:
                scaled_objects = qr_decode(scaled_array)
                logger.info(f"放大{scale}倍后识别结果，找到{len(scaled_objects)}个二维码")
                if scaled_objects:
                    return scaled_objects
            finally:
                del scaled_array  # ✅ 显式删除 numpy 数组
                del scaled_img    # ✅ 显式删除图片对象
        except Exception as e:
            logger.warning(f"图像放大{scale}倍时出错：{str(e)}")
            continue
    
    # 方法 3: 转换为灰度图
    gray_img = img.convert('L')
    gray_array = np.array(gray_img)
    try:
        gray_objects = qr_decode(gray_array)
        logger.info(f"灰度图识别结果数量：{len(gray_objects)}")
        if gray_objects:
            return gray_objects
    finally:
        del gray_array  # ✅ 显式删除 numpy 数组
        del gray_img    # ✅ 显式删除图片对象
    
    # ... 其他方法同样处理 ...
    
    return []
```

---

#### ✅ **方案 3：优化批次大小和数据库提交策略**

**问题根源：** `batch_refresh_status` 函数批次太小且每条记录都 commit

**修复代码：**

```python
# src/routers/ocr/ocr_qr_task_router.py 的 batch_refresh_status 方法
async def batch_refresh_status (
    business_ids: list[str],
    company_ids: list[str],
    province: list[str],
    batch_size: int,
    ai_status: int,
    start_record_id: int,
    db: Session
    ):
    try:
        records = await OCRService.fetch_ocr_records_with_conditions(
            db, business_ids, company_ids, province, start_record_id, 
            batch_size, ai_status, True
        )
        record_ids = []
        
        if not records:
            return record_ids
        
        # ✅ 批量更新，减少 commit 次数
        for idx, record in enumerate(records):
            new_ai_status = _check_result(record.ai_content)
            OCRService.update_ai_status(record.id, new_ai_status, db)
            record_ids.append(record.id)
            
            # ✅ 每 50 条记录提交一次，而不是每条都提交
            if (idx + 1) % 50 == 0:
                db.commit()
        
        # ✅ 最后提交剩余记录
        if len(records) % 50 != 0:
            db.commit()
        
        logger.info(f"分批创建'刷新备案证识别状态'成功，batch_size: {batch_size}, record_ids: {record_ids}")
        return record_ids
        
    except Exception as e:
        logger.error(f"分批创建'刷新备案证识别状态'异常，错误：{str(e)}", exc_info=True)
        db.rollback()  # ✅ 异常时回滚
        raise HTTPException(status_code=500, detail="服务器内部错误")
```

**同时优化批次大小：**

```python
# src/routers/ocr/ocr_qr_task_router.py 的 create_ocr_tasks_refresh_status 方法
@router.post("/tasks/refresh-status", response_model=OCRTaskResponse)
async def create_ocr_tasks_refresh_status(...):
    # ✅ 将批次从 100 提升到 500，减少循环次数
    batch_size = 500  # 原来是 100
    handled_num = 0
    
    batch_record_ids = await batch_refresh_status(
        business_ids,
        company_ids,
        province,
        min(limit, batch_size),  # ✅ 如果 limit 小于 batch_size，则取 limit
        ai_status,
        start_record_id,
        db
    )
    # ... 后续逻辑 ...
```

---

#### ✅ **方案 4：将同步接口改为异步任务（架构优化）**

**问题根源：** `/tasks/refresh-status` 接口同步处理所有记录，内存无法释放

**修复方案：** 将接口改为发布异步任务到 Celery

```python
# src/routers/ocr/ocr_qr_task_router.py
@router.post("/tasks/refresh-status-async", response_model=OCRTaskResponse)
async def create_ocr_tasks_refresh_status_async(
        business_ids: list[str] = Body(..., description="业务 ID 列表"),
        company_ids: list[str] = Body(..., description="公司 ID 列表"),
        province: list[str] = Body(..., description="省份"),
        start_record_id: int = Body(0, description="起始记录 ID"),
        limit: int = Body(2000, description="单次请求处理上限"),
        ai_status: int | None = None,
        db: Session = Depends(get_db_conn)
):
    """异步刷新备案证识别状态（推荐）"""
    try:
        # ✅ 发布异步任务到 Celery，立即返回
        from src.celery_app.ocr_tasks import refresh_status_batch_task
        task = refresh_status_batch_task.delay(
            business_ids=business_ids,
            company_ids=company_ids,
            province=province,
            start_record_id=start_record_id,
            limit=limit,
            ai_status=ai_status
        )
        
        return OCRTaskResponse(
            message="异步任务已提交",
            data=OCRTaskData(
                count=0,
                business_ids=business_ids,
                company_ids=company_ids,
                task_ids=[task.id]
            )
        )
    except Exception as e:
        logger.error(f"异步任务提交失败：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误")
```

**Celery 任务实现：**

```python
# src/celery_app/ocr_tasks.py
@shared_task(bind=True, max_retries=3)
def refresh_status_batch_task(self, business_ids, company_ids, province, 
                               start_record_id, limit, ai_status):
    """异步批量刷新状态"""
    try:
        db = next(get_db_conn())
        batch_size = 500
        handled_num = 0
        
        while handled_num < limit:
            current_batch_size = min(batch_size, limit - handled_num)
            record_ids = batch_refresh_status(
                business_ids, company_ids, province,
                current_batch_size, ai_status, start_record_id, db
            )
            
            if not record_ids:
                break
                
            handled_num += len(record_ids)
            start_record_id = record_ids[-1]
            
            # ✅ 每批次之间短暂休眠，避免内存累积
            import time
            time.sleep(0.1)
        
        return {"status": "success", "handled_num": handled_num}
        
    finally:
        if db:
            db.close()
```

---

#### ✅ **方案 5：增加内存限制和监控**

**在 Docker 配置中限制内存使用：**

```yaml
# compose.yml
services:
  web:
    image: your-image
    deploy:
      resources:
        limits:
          memory: 2G  # ✅ 限制最大内存使用
        reservations:
          memory: 1G
```

**添加内存监控：**

```python
# src/middlewares/memory_monitor.py
import psutil
import logging

logger = logging.getLogger(__name__)

class MemoryMonitor:
    @staticmethod
    def check_memory_usage(threshold_percent=80):
        """检查内存使用率"""
        memory = psutil.virtual_memory()
        if memory.percent > threshold_percent:
            logger.warning(f"内存使用率过高：{memory.percent}%")
            return False
        return True
    
    @staticmethod
    def log_memory_info():
        """记录内存信息"""
        memory = psutil.virtual_memory()
        logger.info(f"内存使用：{memory.percent}%, "
                   f"可用：{memory.available / 1024 / 1024:.2f}MB")
```

---

### 📋 **实施优先级**

| 优先级 | 方案 | 预计效果 | 实施难度 |
|--------|------|----------|----------|
| 🔴 P0 | 方案 1：优化数据库连接管理 | 解决连接池耗尽问题 | ⭐ |
| 🔴 P0 | 方案 2：优化内存管理 | 减少内存泄漏 | ⭐⭐ |
| 🟠 P1 | 方案 3：优化批次大小和提交策略 | 减少数据库 IO | ⭐ |
| 🟡 P2 | 方案 4：异步化改造 | 彻底解决内存累积 | ⭐⭐⭐ |
| 🟢 P3 | 方案 5：内存监控 | 提前预警 | ⭐ |

---

### 🧪 **测试验证**

修复后，使用相同参数测试：

```bash
curl -X POST "http://localhost:8000/api/ocr/qr/tasks/refresh-data" \
  -H "Content-Type: application/json" \
  -d '{
    "business_ids": [],
    "company_ids": [],
    "province": ["1130000"],
    "start_record_id": 0,
    "limit": 7000,
    "handler_type": 1,
    "created_archival": 0,
    "max_update_time": "2026-03-10"
  }'
```

**监控指标：**
1. 内存使用率应稳定在 80% 以下
2. 数据库连接数应保持稳定
3. 服务器不应重启
4. 任务处理完成后内存应回落到基线水平

---

### 📝 **总结**

**根本原因：**
1. 数据库连接未正确释放导致连接池耗尽
2. numpy 数组未显式删除导致内存泄漏
3. 批次太小导致循环次数过多
4. 同步处理大量数据导致内存累积

**核心解决方案：**
1. ✅ 正确关闭数据库连接生成器
2. ✅ 显式删除 numpy 数组
3. ✅ 批量提交减少数据库 IO
4. ✅ 异步化改造（长期方案）

    