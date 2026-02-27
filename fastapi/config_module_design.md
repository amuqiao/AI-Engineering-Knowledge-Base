# 配置模块设计文档

## 一、设计概述

本设计文档描述了基于 `pydantic_settings` 的配置管理模块，用于统一管理应用程序的所有配置，包括数据库配置、日志配置、缓存配置等。该设计遵循**单一职责原则**和**开闭原则**，支持多环境配置、类型安全和配置验证，便于维护和扩展。

## 二、核心设计原则

1. **类型安全**：使用 Pydantic 进行配置验证和类型转换，确保配置的完整性和正确性
2. **集中管理**：所有配置集中在一个模块中，便于维护和监控
3. **环境隔离**：支持多环境配置（开发、测试、生产），通过环境变量实现环境切换
4. **灵活扩展**：支持动态添加新的配置项，无需修改核心代码
5. **配置注入**：通过依赖注入方式访问配置，便于测试和替换
6. **分层设计**：将配置分为不同的层次，便于管理和使用

## 三、目录结构设计

```
fastapi_mvc/
├── app/
│   ├── __init__.py
│   ├── config/              # 配置模块
│   │   ├── __init__.py      # 配置导出和依赖注入
│   │   ├── settings.py      # 主配置类
│   │   ├── database.py      # 数据库配置
│   │   ├── redis.py         # Redis缓存配置
│   │   ├── logging.py       # 日志配置
│   │   ├── security.py      # 安全配置
│   │   └── cors.py          # CORS配置
│   └── ...
├── .env                     # 开发环境变量
├── .env.test                # 测试环境变量
├── .env.prod                # 生产环境变量
└── pyproject.toml           # 项目配置
```

## 四、配置模块实现

### 1. 总配置类

```python
# app/config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
from .database import MySQLConfig, OracleConfig, ClickHouseConfig, SQLiteConfig
from .redis import RedisConfig
from .logging import LoggingConfig
from .security import SecurityConfig
from .cors import CORSConfig

class AppSettings(BaseSettings):
    """应用程序主配置类 - 整合所有配置"""
    
    # 应用基本信息
    APP_NAME: str = "FastAPI MVC"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "基于FastAPI的MVC框架"
    
    # 运行环境
    ENVIRONMENT: str = "development"  # development, test, production
    DEBUG: bool = True
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # API配置
    API_V1_STR: str = "/api/v1"
    API_PREFIX: str = ""
    
    # 配置嵌套 - 整合所有子配置
    DATABASE: Dict[str, Any] = {
        "mysql": MySQLConfig(),
        "oracle": OracleConfig(),
        "clickhouse": ClickHouseConfig(),
        "sqlite": SQLiteConfig()
    }
    
    REDIS: RedisConfig = RedisConfig()
    LOGGING: LoggingConfig = LoggingConfig()
    SECURITY: SecurityConfig = SecurityConfig()
    CORS: CORSConfig = CORSConfig()
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"
        env_nested_delimiter = "__"  # 支持嵌套配置，如 DATABASE__MYSQL__HOST

# 全局主配置实例
app_settings = AppSettings()
```

### 2. 数据库配置

```python
# app/config/database.py
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
from app.config.settings import app_settings

class MySQLConfig(BaseSettings):
    """MySQL数据库配置"""
    HOST: str = "localhost"
    PORT: int = 3306
    USER: str = "root"
    PASSWORD: str = "password"
    DATABASE: str = "fastapi_mvc"
    CHARSET: str = "utf8mb4"
    
    # 连接池配置
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_PRE_PING: bool = True
    POOL_RECYCLE: int = 3600
    
    # 连接超时
    CONNECT_TIMEOUT: int = 10
    READ_TIMEOUT: int = 30
    WRITE_TIMEOUT: int = 30
    
    @property
    def URL(self) -> str:
        """生成MySQL连接URL"""
        return f"mysql+pymysql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}?charset={self.CHARSET}"
    
    @property
    def ENGINE_OPTIONS(self) -> Dict[str, Any]:
        """生成SQLAlchemy引擎选项"""
        return {
            "pool_size": self.POOL_SIZE,
            "max_overflow": self.MAX_OVERFLOW,
            "pool_pre_ping": self.POOL_PRE_PING,
            "pool_recycle": self.POOL_RECYCLE,
            "connect_args": {
                "connect_timeout": self.CONNECT_TIMEOUT,
                "read_timeout": self.READ_TIMEOUT,
                "write_timeout": self.WRITE_TIMEOUT
            }
        }
    
    class Config:
        env_prefix = "MYSQL_"
        env_file = ".env"
        case_sensitive = True

class OracleConfig(BaseSettings):
    """Oracle数据库配置"""
    HOST: str = "localhost"
    PORT: int = 1521
    USER: str = "sys"
    PASSWORD: str = "password"
    SERVICE_NAME: str = "ORCLPDB1"
    
    @property
    def URL(self) -> str:
        """生成Oracle连接URL"""
        return f"oracle+cx_oracle://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/?service_name={self.SERVICE_NAME}"
    
    class Config:
        env_prefix = "ORACLE_"
        env_file = ".env"
        case_sensitive = True

class ClickHouseConfig(BaseSettings):
    """ClickHouse数据库配置"""
    HOST: str = "localhost"
    PORT: int = 8123
    USER: str = "default"
    PASSWORD: str = ""
    DATABASE: str = "default"
    
    @property
    def URL(self) -> str:
        """生成ClickHouse连接URL"""
        return f"clickhouse+http://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}"
    
    class Config:
        env_prefix = "CLICKHOUSE_"
        env_file = ".env"
        case_sensitive = True

class SQLiteConfig(BaseSettings):
    """SQLite数据库配置"""
    DATABASE_FILE: str = "app.db"
    
    @property
    def URL(self) -> str:
        """生成SQLite连接URL"""
        return f"sqlite:///{self.DATABASE_FILE}"
    
    class Config:
        env_prefix = "SQLITE_"
        env_file = ".env"
        case_sensitive = True

# 数据库配置实例
mysql_config = MySQLConfig()
oracle_config = OracleConfig()
clickhouse_config = ClickHouseConfig()
sqlite_config = SQLiteConfig()
```

### 3. Redis配置

```python
# app/config/redis.py
from pydantic_settings import BaseSettings
from typing import Optional

class RedisConfig(BaseSettings):
    """Redis缓存配置"""
    HOST: str = "localhost"
    PORT: int = 6379
    PASSWORD: Optional[str] = None
    DB: int = 0
    PREFIX: str = "fastapi_mvc"
    
    # 连接池配置
    POOL_SIZE: int = 10
    MAX_CONNECTIONS: int = 50
    
    # 超时配置
    CONNECT_TIMEOUT: int = 5
    READ_TIMEOUT: int = 30
    WRITE_TIMEOUT: int = 30
    
    # 缓存过期时间（秒）
    DEFAULT_EXPIRE: int = 3600
    SHORT_EXPIRE: int = 600
    LONG_EXPIRE: int = 86400
    
    @property
    def URL(self) -> str:
        """生成Redis连接URL"""
        if self.PASSWORD:
            return f"redis://:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DB}"
        return f"redis://{self.HOST}:{self.PORT}/{self.DB}"
    
    class Config:
        env_prefix = "REDIS_"
        env_file = ".env"
        case_sensitive = True

# Redis配置实例
redis_config = RedisConfig()
```

### 4. 日志配置

```python
# app/config/logging.py
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any

class LoggingConfig(BaseSettings):
    """日志配置"""
    # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    LEVEL: str = "INFO"
    
    # 日志格式
    FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # 文件日志配置
    FILE_ENABLED: bool = True
    FILE_PATH: str = "logs/app.log"
    FILE_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    FILE_BACKUP_COUNT: int = 5
    
    # 控制台日志配置
    CONSOLE_ENABLED: bool = True
    
    # JSON日志配置（生产环境使用）
    JSON_ENABLED: bool = False
    
    @property
    def DICT_CONFIG(self) -> Dict[str, Any]:
        """生成Python日志配置字典"""
        handlers = []
        
        if self.CONSOLE_ENABLED:
            handlers.append("console")
        
        if self.FILE_ENABLED:
            handlers.append("file")
        
        if self.JSON_ENABLED:
            handlers.append("json")
        
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.FORMAT,
                    "datefmt": self.DATE_FORMAT
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                    "datefmt": self.DATE_FORMAT
                }
            },
            "handlers": {
                "console": {
                    "level": self.LEVEL,
                    "class": "logging.StreamHandler",
                    "formatter": "standard"
                },
                "file": {
                    "level": self.LEVEL,
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": self.FILE_PATH,
                    "maxBytes": self.FILE_MAX_SIZE,
                    "backupCount": self.FILE_BACKUP_COUNT,
                    "formatter": "standard"
                },
                "json": {
                    "level": self.LEVEL,
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": f"{self.FILE_PATH}.json",
                    "maxBytes": self.FILE_MAX_SIZE,
                    "backupCount": self.FILE_BACKUP_COUNT,
                    "formatter": "json"
                }
            },
            "loggers": {
                "": {
                    "handlers": handlers,
                    "level": self.LEVEL,
                    "propagate": True
                },
                "uvicorn": {
                    "handlers": handlers,
                    "level": self.LEVEL,
                    "propagate": False
                },
                "sqlalchemy": {
                    "handlers": handlers,
                    "level": "WARNING",  # SQLAlchemy日志级别
                    "propagate": False
                }
            }
        }
    
    class Config:
        env_prefix = "LOGGING_"
        env_file = ".env"
        case_sensitive = True

# 日志配置实例
logging_config = LoggingConfig()
```

### 5. 安全配置

```python
# app/config/security.py
from pydantic_settings import BaseSettings
from typing import Optional, List

class SecurityConfig(BaseSettings):
    """安全配置"""
    # JWT配置
    SECRET_KEY: str = "your-secret-key-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # 密码配置
    PASSWORD_HASH_ALGORITHM: str = "bcrypt"
    PASSWORD_SALT_LENGTH: int = 12
    
    # CORS配置
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # 速率限制配置
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_AUTHENTICATED: str = "1000/minute"
    
    # CSRF配置
    CSRF_ENABLED: bool = True
    CSRF_SECRET_KEY: str = "your-csrf-secret-key-change-me"
    
    class Config:
        env_prefix = "SECURITY_"
        env_file = ".env"
        case_sensitive = True

# 安全配置实例
security_config = SecurityConfig()
```

### 6. CORS配置

```python
# app/config/cors.py
from pydantic_settings import BaseSettings
from typing import List

class CORSConfig(BaseSettings):
    """CORS配置"""
    ORIGINS: List[str] = ["*"]
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: List[str] = ["*"]
    ALLOW_HEADERS: List[str] = ["*"]
    
    class Config:
        env_prefix = "CORS_"
        env_file = ".env"
        case_sensitive = True

# CORS配置实例
cors_config = CORSConfig()
```

## 五、配置依赖注入

```python
# app/config/__init__.py
from fastapi import Depends
from typing import Type, TypeVar, Optional, Dict, Any
from functools import lru_cache

# 导入配置类和实例
from app.config.settings import AppSettings
from app.config.database import MySQLConfig, OracleConfig, ClickHouseConfig, SQLiteConfig
from app.config.redis import RedisConfig
from app.config.logging import LoggingConfig
from app.config.security import SecurityConfig
from app.config.cors import CORSConfig

# 配置类型变量
T = TypeVar('T')

@lru_cache()
def get_settings() -> AppSettings:
    """获取总配置实例 - 使用lru_cache确保单例"""
    return AppSettings()

class ConfigDeps:
    """配置依赖注入容器 - 支持从总配置访问所有子配置"""
    
    @staticmethod
    def settings():
        """获取总配置依赖注入"""
        return Depends(get_settings)
    
    @staticmethod
    def app():
        """获取应用基本配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s)
    
    @staticmethod
    def mysql():
        """获取MySQL配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s.DATABASE['mysql'])
    
    @staticmethod
    def oracle():
        """获取Oracle配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s.DATABASE['oracle'])
    
    @staticmethod
    def clickhouse():
        """获取ClickHouse配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s.DATABASE['clickhouse'])
    
    @staticmethod
    def sqlite():
        """获取SQLite配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s.DATABASE['sqlite'])
    
    @staticmethod
    def redis():
        """获取Redis配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s.REDIS)
    
    @staticmethod
    def logging():
        """获取日志配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s.LOGGING)
    
    @staticmethod
    def security():
        """获取安全配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s.SECURITY)
    
    @staticmethod
    def cors():
        """获取CORS配置依赖注入"""
        return Depends(lambda s=Depends(get_settings): s.CORS)
    
    @staticmethod
    def database(db_type: str = 'mysql'):
        """动态获取指定类型的数据库配置"""
        def get_db_config(settings: AppSettings = Depends(get_settings)) -> Any:
            if db_type not in settings.DATABASE:
                raise ValueError(f"Database type '{db_type}' not configured")
            return settings.DATABASE[db_type]
        return Depends(get_db_config)

# 导出配置实例和依赖注入函数
config_deps = ConfigDeps()

# 导出配置实例和依赖注入
__all__ = [
    "get_settings",
    "config_deps",
    "AppSettings",
    "MySQLConfig",
    "OracleConfig",
    "ClickHouseConfig",
    "SQLiteConfig",
    "RedisConfig",
    "LoggingConfig",
    "SecurityConfig",
    "CORSConfig"
]

# 直接导出总配置实例（方便直接使用）
settings = get_settings()
__all__.append("settings")
```

## 六、环境变量配置示例

### 1. 开发环境 (.env)

```env
# 应用配置
APP_NAME=FastAPI MVC
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=True

# MySQL配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=fastapi_mvc
MYSQL_POOL_SIZE=10
MYSQL_MAX_OVERFLOW=20

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PREFIX=fastapi_mvc

# 日志配置
LOGGING_LEVEL=DEBUG
LOGGING_FILE_ENABLED=True
LOGGING_FILE_PATH=logs/app.log

# 安全配置
SECURITY_SECRET_KEY=your-secret-key-change-me
SECURITY_ALGORITHM=HS256
SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS配置
CORS_ORIGINS=*
```

### 2. 生产环境 (.env.prod)

```env
# 应用配置
APP_NAME=FastAPI MVC
APP_VERSION=1.0.0
ENVIRONMENT=production
DEBUG=False

# MySQL配置
MYSQL_HOST=mysql-server
MYSQL_PORT=3306
MYSQL_USER=app_user
MYSQL_PASSWORD=${MYSQL_PASSWORD}  # 从环境变量获取
MYSQL_DATABASE=fastapi_mvc
MYSQL_POOL_SIZE=20
MYSQL_MAX_OVERFLOW=50

# Redis配置
REDIS_HOST=redis-server
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}  # 从环境变量获取
REDIS_DB=0
REDIS_PREFIX=fastapi_mvc

# 日志配置
LOGGING_LEVEL=INFO
LOGGING_FILE_ENABLED=True
LOGGING_FILE_PATH=/var/log/fastapi/app.log
LOGGING_JSON_ENABLED=True

# 安全配置
SECURITY_SECRET_KEY=${SECRET_KEY}  # 从环境变量获取
SECURITY_ALGORITHM=HS256
SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES=60

# CORS配置
CORS_ORIGINS=https://example.com,https://api.example.com
```

## 七、配置使用示例

### 1. 直接使用配置

```python
from app.config import app_settings, mysql_config

# 使用应用配置
print(f"应用名称: {app_settings.APP_NAME}")
print(f"应用版本: {app_settings.APP_VERSION}")

# 使用数据库配置
print(f"MySQL主机: {mysql_config.HOST}")
print(f"MySQL连接URL: {mysql_config.URL}")
```

### 2. 依赖注入使用配置

```python
from fastapi import Depends, APIRouter
from app.config import ConfigDeps, MySQLConfig, RedisConfig

router = APIRouter()

@router.get("/config")
def get_config(
    mysql_config: MySQLConfig = Depends(ConfigDeps.mysql),
    redis_config: RedisConfig = Depends(ConfigDeps.redis)
):
    """获取配置信息"""
    return {
        "mysql": {
            "host": mysql_config.HOST,
            "port": mysql_config.PORT
        },
        "redis": {
            "host": redis_config.HOST,
            "port": redis_config.PORT
        }
    }
```

### 3. 在服务层使用配置

```python
from app.config import config_deps
from app.config.security import SecurityConfig

class UserService:
    """用户服务"""
    
    def __init__(self,
                 security_config: SecurityConfig = config_deps.security()):
        self.security_config = security_config
    
    def create_access_token(self, user_id: int):
        """创建访问令牌"""
        # 使用安全配置
        expire_minutes = self.security_config.ACCESS_TOKEN_EXPIRE_MINUTES
        algorithm = self.security_config.ALGORITHM
        # ... 令牌生成逻辑
```

### 4. 日志配置使用

```python
# app/main.py
import logging.config
from fastapi import FastAPI
from app.config import logging_config, app_settings

# 配置日志
logging.config.dictConfig(logging_config.DICT_CONFIG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=app_settings.APP_NAME,
    version=app_settings.APP_VERSION,
    debug=app_settings.DEBUG
)

@app.on_event("startup")
async def startup_event():
    logger.info(f"应用启动: {app_settings.APP_NAME} v{app_settings.APP_VERSION}")
    logger.info(f"环境: {app_settings.ENVIRONMENT}")
```

## 八、配置管理最佳实践

1. **使用环境变量存储敏感信息**：避免将密码、密钥等敏感信息硬编码到代码中
2. **分层配置**：将不同类型的配置分开管理，便于维护
3. **默认值设置**：为所有配置项设置合理的默认值，便于开发和测试
4. **类型安全**：使用Pydantic的类型注解确保配置类型正确
5. **配置验证**：利用Pydantic的验证功能确保配置完整性和正确性
6. **多环境支持**：为不同环境（开发、测试、生产）提供不同的配置文件
7. **配置注入**：通过依赖注入方式访问配置，便于测试和替换
8. **配置文档**：为每个配置项添加详细的文档，说明其用途和默认值
9. **配置监控**：考虑添加配置监控功能，便于观察配置的变化
10. **配置热更新**：对于需要动态更新的配置，考虑实现配置热更新机制

## 九、扩展性设计

### 1. 添加新的配置项

要添加新的配置项，只需在相应的配置类中添加字段即可：

```python
class AppSettings(BaseSettings):
    # 现有配置
    APP_NAME: str = "FastAPI MVC"
    
    # 新添加的配置
    NEW_CONFIG_ITEM: str = "default-value"
```

### 2. 添加新的配置类

要添加新的配置类型，只需创建新的配置类并添加到依赖注入容器中：

```python
# app/config/new_config.py
from pydantic_settings import BaseSettings

class NewConfig(BaseSettings):
    """新配置类型"""
    NEW_SETTING: str = "default"
    
    class Config:
        env_prefix = "NEW_"
        env_file = ".env"

# 创建配置实例
new_config = NewConfig()
```

然后在`app/config/__init__.py`中添加到依赖注入容器：

```python
from app.config.new_config import new_config, NewConfig

class ConfigDeps:
    # 现有配置方法
    
    @staticmethod
    def new() -> NewConfig:
        """获取新配置"""
        return new_config
    
    @classmethod
    def get(cls, config_type: Type[T]) -> T:
        config_mapping = {
            # 现有配置映射
            NewConfig: cls.new()
        }
        # ...
```

## 十、总结

本设计文档描述了基于`pydantic_settings`的配置管理模块，该模块具有以下特点：

1. **类型安全**：使用Pydantic进行配置验证和类型转换
2. **集中管理**：所有配置集中在一个模块中，便于维护
3. **环境隔离**：支持多环境配置，通过环境变量实现环境切换
4. **灵活扩展**：支持动态添加新的配置项和配置类型
5. **配置注入**：通过依赖注入方式访问配置，便于测试和替换
6. **分层设计**：将配置分为不同的层次，便于管理和使用
7. **生产级特性**：支持日志配置、安全配置、CORS配置等生产级特性

该配置模块设计遵循了良好的设计原则，具有良好的可扩展性和可维护性，适合用于中大型FastAPI应用程序。