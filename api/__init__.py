"""
api —— HTTP 服务层（FastAPI）

本包把 pipeline_offline / pipeline_online 的能力包装成常驻服务，供班组终端、
企业微信机器人、内部系统通过 HTTP 调用。

    config.py   配置加载（base + 部署覆盖 + 环境变量，配置与镜像分离）
    schemas.py  请求 / 响应模型（对外契约）
    service.py  用例编排（问答 / 检索 / 入湖 / 反馈）
    main.py     FastAPI 应用与路由

启动方式见 main.py 顶部说明。
"""
