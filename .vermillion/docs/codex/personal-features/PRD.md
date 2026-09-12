# 个人功能集成

## 要做什么

在上游稳定版本上维护独立的 feature 分支，并逐个合入版本化 integration 分支。个人功能包括：

- exec-argv：以参数数组调用原生可执行程序，在审批、hook、后台会话和输出中保留工具身份及原始参数。
- retry-empty-final-answer：常规回合缺少最终回答或最终回答为空时，最多补试一次。
- subagent-identity-labels：向子代理提供身份事实和继承历史边界，保持上游工具协议。
- model-aware-compaction：根据 provider 与当前模型的能力选择远端压缩或本地摘要。
- local-compaction-handoff：本地摘要保留决策、工作进度和关键证据，并提供原始 rollout 查阅路径。
- large-request-upload-resilience：检测未完成的大请求上传异常，取消该次上传并通过新连接补试一次。
- auth-independent-tools：已安装的生图工具满足功能开关、模型要求和套餐条件时，任何 provider 均可向模型暴露该工具，不按 provider 类型、声明的生图能力或登录方式隐藏；实际服务请求沿用原有凭据和授权流程。[用户：确保任何provider都能看到生图]

## 明确不做什么

个人功能集合及新版本迁移 skill 不包含 chattree。[用户：在新版本上移除skill里chattree的部分，只迁移其他几个ft。]

本项工具可见性调整不涉及 Apps 与 History／Notes。[用户：另外两个还是先不管]

## 已定的实现方向

exec-argv 的工具说明区分原生程序与 shell 脚本。管道、重定向、通配符展开、shell 内建命令、初始化逻辑及 Windows 脚本启动器使用对应 shell。完整路径解决程序查找问题，不改变脚本的解释语义；显式启动的解释器仍会解析自己的代码参数。

生图可见性在扩展注册和每轮工具规划中统一放行 provider，保留功能开关、模型图像模态和套餐条件。它不生成未安装的工具，也不改变后端认证或服务权限；工具可见不表示当前 provider 已实现生图接口。

各功能的完整行为契约及迁移约束由 [.agents/skills/personal-features-port](../../../../.agents/skills/personal-features-port/SKILL.md) 维护。发布打包流程归 integration 分支所有。
