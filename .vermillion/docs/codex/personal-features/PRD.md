# 个人功能集成

## 要做什么

在上游稳定版本上维护独立的 feature 分支，并逐个合入版本化 integration 分支。个人功能包括：

- exec-argv：以参数数组调用原生可执行程序，在审批、hook、后台会话和输出中保留工具身份及原始参数。
- retry-empty-final-answer：常规回合缺少最终回答或最终回答为空时，最多补试一次。
- subagent-identity-labels：向子代理提供身份事实和继承历史边界，保持上游工具协议。
- subagent-continuation：子任务完成后自动续跑，等待子任务期间及时处理 steer。完整行为与边界见[子任务续跑契约](../../../../.agents/skills/personal-features-port/references/feature-subagent-continuation.md)。
- model-aware-compaction：根据 provider 与当前模型的能力选择远端压缩或本地摘要。
- local-context-compaction：后台并行标记工具记录，达到可配置的释放差额后统一清理，保留对话原文与原始记录回查。未标记的工具内容积累到窗口的 5%（可配置）才发起一批标记，不按条数触发；清理默认要求释放窗口的 30%。清理时同时裁剪之前回合的推理：当前回合的推理始终保留，更早的推理只保留最近的、合计不超过窗口 5%（可配置）的部分；本身就不回传旧推理的模型不受影响。带截图的工具输出也会被标记，清理时去掉图片并注明张数；参数较大的工具调用（如补丁）随结果一起改为一行简述，调用记录本身保留。回查原始记录时，模型能直接看到原图。每批优先标记体积最大的工具输出，按预计响应大小装批，零碎小输出可一批标完；标记请求原样沿用正常请求的前缀，可命中提示缓存。判为保留的输出只在当次用户回合内有效，用户再发消息后会随下一批重新判断；精简和删除的结论长期有效。标记、工具清理和推理裁剪对所有模型生效，不论完整压缩走哪条路线。到 Codex 原有压缩条件或手动压缩时，provider 与当前模型都支持远端压缩就走远端压缩，否则做分级压缩：当前窗口按上述规则清理后原文保留，成为下一次压缩时的上一窗口；上一窗口连同已有摘要由模型合并成一份新的交接摘要（保留用户目标、进展、关键决定、约束和后续步骤）。指令类内容（AGENTS.md、环境、技能说明等）每类只保留最新一份，旧副本直接丢弃。清理后的当前窗口超过窗口的 50%（可配置）时，它最早的部分也并入摘要，因此反复压缩后占用仍然收敛。校验通过的标记立即落盘，恢复会话或从中 fork 出的新会话都继续有效，不会重复标记。app-server 的 `thread/toolCleanup/read` 返回当前标记预计可释放的上下文量、自动清理要求的释放量以及是否有标记请求在途；`thread/toolCleanup/apply` 忽略释放要求，立即应用已有标记并返回实际释放量，不等待在途标记、不发模型请求。完整契约见[本地上下文压缩](../../../../.agents/skills/personal-features-port/references/feature-local-context-compaction.md)。
- large-request-upload-resilience：检测未完成的大请求上传异常，取消该次上传并通过新连接补试一次。
- stream-retry-timeout：响应长时间没有完整结果时保留原请求并补发，采用先就绪的一条，规则见[功能契约](../../../../.agents/skills/personal-features-port/references/feature-stream-retry-timeout.md)。
- auth-independent-tools：已安装的生图工具满足功能开关、模型要求和套餐条件时，任何 provider 均可向模型暴露该工具，不按 provider 类型、声明的生图能力或登录方式隐藏；实际服务请求沿用原有凭据和授权流程。[用户：确保任何provider都能看到生图]

integration 分支每次推送后自动发布 Windows x64 和 macOS Apple Silicon 便携包，作为仓库的 latest release。macOS 包只有 ad-hoc 签名、未经 Apple 公证：用 `gh release download` 或 `curl` 下载的包解压后可直接运行，浏览器下载的包首次运行前需要清除隔离属性。

## 明确不做什么

个人功能集合及新版本迁移 skill 不包含 chattree。[用户：在新版本上移除skill里chattree的部分，只迁移其他几个ft。]

本项工具可见性调整不涉及 Apps 与 History／Notes。[用户：另外两个还是先不管]

## 已定的实现方向

exec-argv 的工具说明区分原生程序与 shell 脚本。管道、重定向、通配符展开、shell 内建命令、初始化逻辑及 Windows 脚本启动器使用对应 shell。完整路径解决程序查找问题，不改变脚本的解释语义；显式启动的解释器仍会解析自己的代码参数。

生图可见性在扩展注册和每轮工具规划中统一放行 provider，保留功能开关、模型图像模态和套餐条件。它不生成未安装的工具，也不改变后端认证或服务权限；工具可见不表示当前 provider 已实现生图接口。

各功能的完整行为契约及迁移约束由 [.agents/skills/personal-features-port](../../../../.agents/skills/personal-features-port/SKILL.md) 维护。发布打包流程归 integration 分支所有。
