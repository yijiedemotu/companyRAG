# 数据库迁移规范（Alembic）

> 这份文档记录的是**这个项目里真实踩过的坑**，不是 Alembic 官方文档的复述。
> 权威的模型定义在 `backend/src/knowflow/db/models/`，权威的配置在 `backend/src/knowflow/core/config.py`。

---

## 1. 常用命令

```powershell
cd backend

# 升级到最新
..\.venv\Scripts\python.exe -m alembic upgrade head

# 回退一个版本
..\.venv\Scripts\python.exe -m alembic downgrade -1

# 生成迁移（改完模型之后）
..\.venv\Scripts\python.exe -m alembic revision --autogenerate -m "add xxx field"

# 查看当前版本 / 历史
..\.venv\Scripts\python.exe -m alembic current
..\.venv\Scripts\python.exe -m alembic history --verbose

# 只想看将要执行的 SQL（不连库、不改数据）—— 上线前 review 用
..\.venv\Scripts\python.exe -m alembic upgrade head --sql
```

**指定数据库**（CI / 测试用，不改 `.env`）：

```powershell
# ⚠ `-x` 必须放在**子命令之前**，否则报 `unrecognized arguments: -x`
..\.venv\Scripts\python.exe -m alembic -x "db_url=mysql+pymysql://root:1234@127.0.0.1:3306/knowflow_test?charset=utf8mb4" upgrade head
```

---

## 2. 四个必须知道的坑

### 坑 1：`alembic.ini` 必须是**纯 ASCII**

Alembic 用 `configparser` 读 `alembic.ini`，而它按**系统 locale 编码**打开文件
（`read(file, encoding="locale")`）。在中文 Windows 上 locale 是 **GBK**，
所以只要这个文件里有**一个中文字符**，就会：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0x82 in position 18: illegal multibyte sequence
```

而且**报错信息完全指不到 `alembic.ini`**（栈里全是 configparser 内部帧），
第一次遇到会以为是代码问题，能查很久。

**规则**：`backend/alembic.ini` 里的注释只能写英文。
中文说明写在这里，或者写进 `backend/alembic/env.py`（Python 源码永远是 UTF-8）。

### 坑 2：`models/__init__.py` 必须 import 所有模型

`alembic/env.py` 里 `target_metadata = Base.metadata`，
而 `Base.metadata` **只认识"已经被导入过"的模型类**。

如果你新加了一个模型文件却忘了在 `backend/src/knowflow/db/models/__init__.py` 里 import 它，
autogenerate 会认为"数据库里那张表是多余的"，于是生成一条：

```python
op.drop_table('your_new_table')     # ← 灾难
```

**所以每加一个模型文件，必须去 `models/__init__.py` 补一行 import。**
`env.py` 里已经有一条注释在提醒这件事。

### 坑 3：`compare_type=True` 不能省

MySQL 上把 `VARCHAR(64)` 改成 `VARCHAR(128)`，如果 `compare_type=False`，
autogenerate **检测不到差异**，迁移里就不会有 `ALTER`，改了模型却以为迁移写好了。

`env.py` 里已经设了：

```python
context.configure(..., compare_type=True, compare_server_default=True)
```

### 坑 4：autogenerate 只写"结构差异"，不写"数据迁移"

重命名字段会被识别成「删一个 + 加一个」，**数据会丢**。
正确做法：手写 `op.alter_column(..., new_column_name=...)`，或者
加新列 → 写数据迁移 `op.execute("UPDATE ...")` → 删旧列。

---

## 3. 写迁移的规范

1. **`upgrade()` 与 `downgrade()` 都要写**。只写 `upgrade` 的迁移在回滚时是坏的 ——
   而回滚恰恰是你最需要它工作的时刻。（`pass` 也算没写，除非确实不可逆并写清原因。）
2. **迁移文件头部写清"它干了什么"**。`script.py.mako` 里已经留了模板：
   ```
   迁移说明（**每个迁移都必须手写这段，否则三个月后没人知道它干了什么**）：
     -
   ```
3. **加非空列必须给默认值**，否则已有数据行会导致 `ALTER` 失败：
   ```python
   op.add_column("documents", sa.Column("token_count", sa.Integer(), nullable=False,
                                        server_default="0"))
   # 之后如果不想保留默认值：
   op.alter_column("documents", "token_count", server_default=None)
   ```
4. **索引要显式命名**。项目在 `db/base.py` 里配了 `NAMING_CONVENTION`，
   所以 autogenerate 出来的名字是可预测的（`ix_<表>_<列>`、`uq_<表>_<列>`、`fk_<表>_<列>_<目标表>`），
   不要手动改成别的名字，否则 `DROP` 会找不到目标。
5. **大表加索引要评估锁表**。MySQL 8 的 `ALTER` 多数支持 online DDL，但并非全部；
   数据量大时用 `ALGORITHM=INPLACE, LOCK=NONE` 并先在从库演练。

---

## 4. 模型与迁移必须一致（CI 有门禁）

CI（`.github/workflows/ci.yml`）里有一步 `Check for model/migration drift`：
跑一次 `alembic revision --autogenerate`，如果生成的迁移文件里出现 `op.` 调用，
就判定"模型改了但迁移没写"并让 CI 失败。

本仓库当前状态（实测）：

```
[4] Alembic 迁移与模型零 drift
    upgrade head 建出 15 张表（14 张业务表 + alembic_version）
    autogenerate 没有打印任何 "Detected" 行
    生成的临时文件里 upgrade/downgrade 都是 pass
    op. 调用数 = 0   → 模型与迁移完全一致
```

**在本地自查同样的事情**：

```powershell
cd backend
..\.venv\Scripts\python.exe -m alembic revision --autogenerate -m "drift check"
# 看到生成的 versions/*_drift_check.py 里只有 pass 就是一致的
# 确认后删掉它
```

---

## 5. 迁移与"多环境"的关系

`alembic.ini` 里 `sqlalchemy.url` **故意留空**，URL 由 `env.py` 从
`knowflow.core.config.get_settings()` 读。这样做有两个好处：

1. **密码不进 git**（`alembic.ini` 会被提交，`.env` 不会）；
2. **只有一个配置来源** —— 不会出现"改了 `.env` 但 alembic 用的是 ini 里的旧值"。

要临时切库用 `-x db_url=...`（见第 1 节），不要改 ini。

> ⚠ **`DATA_DIR` 与 `DATABASE_URL` 是一一对应的**（见 `docs/00` 第 8 节）：
> 切换数据库时必须**同时**切换 `DATA_DIR`，否则向量库（存在 `DATA_DIR/chroma`）
> 会和新的关系库串起来 —— 表现为"检索结果指向不存在的 chunk"。
> `scripts/smoke_pipeline.py` 就是靠"独立库 + 独立 DATA_DIR"来避免这个问题的。
