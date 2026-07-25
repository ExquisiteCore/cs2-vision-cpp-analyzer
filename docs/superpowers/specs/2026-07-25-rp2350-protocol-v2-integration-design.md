# RP2350 协议 v2 集成加固设计

## 背景

RP2350 固件和 `rp2350_hid_bridge_cpp` SDK 已升级到可靠协议 v2。SDK 当前提交为
`c3d9a55`，协议 v2 增加了 500 ms 心跳、DTR 会话控制、两秒安全租约、同序列重试、
`BUSY` 延迟、扩展 NACK、并发串行化和安全关闭顺序。

现有 GTX 1080 Ti 生产包中的 `vision_runtime.dll` 构建于 2026-07-22，只包含旧 SDK。
当前 C++ 源码可以用最新版 SDK 编译，但只执行 `ping()`，运行时关闭路径可能让串口
异常从析构函数逸出，组包器也只能判断是否编入了 SDK，不能阻止旧协议 DLL 被交付。

## 目标

1. 运行时只接受提供协议 v2、鼠标、可靠重试、安全租约和取消能力的固件。
2. 打开设备时执行 `ping()`、`info()` 和 `caps()` 三项只读健康检查。
3. USB 拔出、串口失效或停止命令失败时，关闭流程仍必须完成且不得终止宿主进程。
4. CMake 和 xmake 都能用最新版仅头文件 SDK 构建；CMake 显式携带线程依赖。
5. SM61 组包器拒绝未编入协议 v2 健康检查的 DLL，并在清单中记录协议 v2 组件。
6. 最终仍固定 GTX 1080 Ti、ORT GPU 1.17.3、TensorRT 8.6.1.6、CUDA 11.8、
   cuDNN 8.9.7 和 FP32，不混入开发机的 ORT 1.24.4。

## 非目标

- 不改变检测、跟踪、头部优先、身体兜底、灵敏度标定或开火策略。
- 不增加 UI、自动重连或无上限重试。
- 不改变现有 C ABI，也不要求 Python 调用方修改方法顺序。
- 不在运行包中捆绑固件刷写工具；生产板已由用户更新固件。

## 方案比较

### 方案 A：只用新 SDK 重新编译

优点是改动最小，旧调用也与新版 SDK 源码兼容。缺点是无法验证固件能力，异常关闭仍
可能终止进程，组包器以后也可能再次打入旧 DLL。因此不采用。

### 方案 B：接入加固并重新打包（采用）

在现有 `HidClient` 边界内增加协议 v2 健康解析和不抛异常的关闭能力；运行时显式查询
设备能力；构建和组包阶段增加协议 v2 门禁。该方案解决当前兼容问题并防止回归，同时
不扩大 C API 和 Python API，风险可控。

### 方案 C：增加完整设备管理与自动重连子系统

可以暴露固件版本、热插拔事件和重连状态，但会引入后台状态机、重复命令判定和新的
C API。当前只需要稳定的单设备会话，方案 C 超出范围。

## 运行时设计

### 协议 v2 健康信息

新增内部 `HidDeviceHealth` 值对象和纯解析函数。解析函数接收 SDK `info()` 与
`caps()` 的字节载荷，并执行以下检查：

- `info` 至少四字节且报告协议版本 2；
- `caps` 至少五字节且报告协议版本 2；
- 16 位能力位必须同时包含鼠标、可靠重试、安全租约和取消；
- 任何缺失都抛出带固定文本 `RP2350 protocol v2 capabilities are required` 的错误。

固定错误文本同时作为发布 DLL 的协议 v2 构建标记。解析函数是纯函数，可在没有板卡
时覆盖正常、截断、旧协议和缺能力场景。

### 连接流程

`Rp2350HidClient` 使用 `HidBridgeOptions` 构造 SDK 客户端，保持默认 115200 波特率、
1000 ms 基础超时、两次重试和 500 ms 心跳。构造顺序为：

1. `open()`，由 SDK 置位 DTR 并启动心跳；
2. `ping()`；
3. `info()`；
4. `caps()`；
5. 解析并保存协议 v2 健康结果。

任何一步失败时由 `HidBridge` RAII 关闭当前会话，运行时不进入标定或物理输出阶段。
不在上层重复最终失败的命令。

### 安全关闭

`HidClient` 增加 `close() noexcept`。具体 RP2350 客户端将其映射到新版 SDK 的
`HidBridge::close() noexcept`。

新增 `close_hid_client_noexcept()` 辅助函数：先尽力调用 `stop_all()`，无论结果如何
都继续调用 `close()`。它允许用记录型测试客户端验证停止失败后仍执行关闭。

`RuntimeSession::~RuntimeSession()` 和 `RuntimeSession::close()` 明确标记为
`noexcept`。关闭顺序为：

1. 立即关闭开火规划并把会话标记为关闭；
2. 销毁引用 HID 客户端的发送器，阻止新命令；
3. 尽力 `stop_all()`，随后无条件 `close()` HID 会话；
4. 尽力释放 DXGI/视频源；
5. 关闭日志并销毁检测、跟踪和分析资源。

显式 `va_stop_all()` 仍保留错误报告能力；`va_close()` 负责最终清理，不因已断开的 USB
再次使宿主崩溃。

## 构建设计

### CMake

- 最低版本提升到 3.20，与 SDK 文档一致；
- 在启用 HID SDK 时 `find_package(Threads REQUIRED)`；
- `vision_analyzer_core` 链接 `Threads::Threads`；
- 编译期断言 SDK `PROTOCOL_VERSION == 2`，旧 SDK 直接构建失败。

### xmake

继续显式传入 `hid_sdk_root`。Windows/MSVC 的 `std::thread` 由运行库提供，不增加额外
系统库；相同的协议版本编译期断言保证 xmake 也不能误用旧头文件。

生产构建必须清理并重新配置为：

```text
onnxruntime_root=D:\Tool\onnxruntime-win-x64-gpu-1.17.3
hid_sdk_root=D:\project\cs2-vision-trainer\tools\rp2350_hid_bridge_cpp
mode=release
```

## 组包设计

组包器读取 `vision_runtime.dll` 后必须找到协议 v2 固定标记；只有 SDK 存在但缺少该
标记时立即失败。生成的 `runtime-manifest.json` 增加组件：

```json
{
  "id": "rp2350-hid-sdk",
  "version": "protocol-v2",
  "sourceMode": "header-only-build"
}
```

生产机静态验证同时要求该组件恰好出现一次且版本正确。这样旧 DLL、旧清单或混合包都
不能通过一键检查。

最终生成新的完整 v2 ZIP，同时从旧包与新包清单差异生成小型覆盖补丁。补丁只包含变化
的不可变文件及新清单，用户无需重新上传全部 CUDA/TensorRT 运行库。

## 测试策略

1. C++ 算法测试先增加协议载荷解析和停止失败仍关闭的失败用例，再实现代码。
2. 编译期检查 `RuntimeSession::close()` 为 `noexcept`。
3. 包工具测试先增加旧 DLL 缺少 v2 标记时必须拒绝、模板验证必须检查 v2 清单组件的
   失败用例，再修改 PowerShell。
4. 强制重建最新版 SDK 并运行其 CTest。
5. 使用 xmake 强制重建 C++ DLL、CLI、算法测试和 C API 测试。
6. 使用 CMake clean build 并运行 CTest，验证线程依赖。
7. 使用 ORT 1.17.3 生成完整 SM61 包；执行清单验证、两轮 TensorRT 视频 dry-run 和
   解压后 smoke test。
8. 无开发机 RP2350 时不声称硬件通过；最终由生产机 COM3 执行只读健康检查和标定。

## 完成标准

- 旧 SDK 无法编译，旧协议 DLL 无法组包；
- 协议 v2 健康载荷通过，缺少任何必需能力时连接失败且不产生 HID 输入；
- `stop_all()` 抛异常时仍调用设备 `close()`，运行时析构不抛异常；
- SDK、xmake、CMake、C API、包工具和生产包验证全部通过；
- 新包确认使用 ORT 1.17.3，而不是 ORT 1.24.4；
- 生产机重新测试时能通过 COM3 的 `ping/info/caps`，随后才进入灵敏度标定。
