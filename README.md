# ⭐⭐**CosyVoiceCpp**⭐⭐

## 项目介绍

对阿里出品[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)模型的Cpp化部署

通过分割原始python项目，按模块导出为onnx模型，摆脱臃肿的python环境，实现在Cpp语言环境下的文字转语音推理。

在您的硬件条件允许的条件下，您可以将本项目嵌入到任何其他Cpp项目中。

## 如何使用

⚠️⚠️⚠️**后续所有终端控制台输入输出基于 UTF-8编码， windows平台在终端键入`chcp 65001`以支持UTF-8**

### 1. 环境配置

本项目仅可在**Windows10/11**平台下完整运行，其他平台的运行需考虑替换所用依赖库的对应平台版本

- 编译环境

  Visual Studio Professional 2022 LTSC 17.0

  后续所用CUDA Toolkit的版本和 VS 版本相关，如果CUDA Toolkit较老，而VS较新，则可能导致CUDA Toolkit安装失败，建议使用17.0及较落后版本的VS。

  微软官方只提供VS的最新社区版或专业版，而这并不适配后续所用CUDA Toolkit 12.1，而对于历史版本，只有专业版提供下载，下载渠道：

  [Visual Studio 2022 发行历史记录 | Microsoft Learn](https://learn.microsoft.com/zh-cn/visualstudio/releases/2022/release-history#uninstalling-visual-studio-to-go-back-to-an-earlier-release)

  LTSC 17.0 在页面底部

- 准备依赖库

  1. [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

  2. [cuDNN 9.3.0](https://developer.nvidia.com/cudnn-archive)

  3. [TensorRT-10.1](https://developer.nvidia.com/tensorrt/download/10x) 

     下载页面中，支持Windows的TensorRT 10.1安装包有两个版本，注意选择名称为**TensorRT 10.1 GA for Windows 10, 11, Server 2019, Server 2022 and CUDA 12.0 to 12.4 ZIP Package**的版本。

     > 仅支持40系及以上显卡

  4. [cuDNN 8.9.7](https://developer.nvidia.com/rdp/cudnn-archive)

     **为什么要下载两个cuDNN?**

     因为TensorRT 10.1 只支持到cuDNN8.9.7， 而后续使用的ONNX Runtime只支持CuDNN9.x。

  5. libtorch

     [libtorch-win-shared-with-deps-2.3.1+cu121](https://download.pytorch.org/libtorch/cu121)

  6. nvToolsExt

     因为libtorch编译需要nvToolsExt，但 [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)并不提供，所以需要下载CUDA11.6，在安装页面仅选择【Nsight NVTX】进行安装，安装结束后，便可在`C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64`下找到nvToolsExt.dll

     <img src=".\images\image-20250801233428979.png" alt="image-20250801233428979" style="zoom:100%;" />

  7. [onnxruntime-win-x64-gpu-1.21.0](https://github.com/microsoft/onnxruntime/releases)

  8. libsndfile1.2.2

     强烈建议在Windows平台使用[vcpkg](https://github.com/microsoft/vcpkg)安装libsndfile库，如果你已经成功安装vcpkg，进入vcpkg安装根目录，使用安装指令：

     ```bash
     vcpkg.exe install libsndfile:x64-windows
     ```

  9. [tokenizer-cpp](https://github.com/mlc-ai/tokenizers-cpp) ✅

  10. [nlohmann-json](https://github.com/nlohmann/json)✅

 - 安装依赖库

   ✅表示该依赖已包含在github仓库中，无需额外下载编译

   1. 变更仓库中的`CMakeLists.txt`

      如果已经成功使用vcpkg安装sndfile，请替换`CMakeLists.txt`中的vcpkg库安装路径

      ```cmake
      set(VCPKG_X64-windows "your vcpkg installed dir")
      For example:
      set(VCPKG_X64-windows "F:/Zlab/C++/vcpkg/installed/x64-windows")
      ```

      如果已经成功安装CUDA Toolkit 12.1，请替换`CMakeLists.txt`中的CUDA Toolkit 路径，一共有3处，如果是默认安装，一般无需变更

      ```cmake
      # 设置CUDA相关路径
      set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
      set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe")
      
      # 设置include目录
      include_directories(
        include
        ${VCPKG_X64-windows}/include
        ${ThirdParty}/include
        ${TORCH_INCLUDE_DIRS}
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
      )
      ```

      

 2. 复制依赖库的 `include` 、`lib` 、`share` 到项目

    项目中包含子目录`ThirdParty`，其结构为：

    ```bash
    ThirdParty/
    |--- inclue
    |--- lib
    |--- share
    ```

    将[TensorRT-10.1](https://developer.nvidia.com/tensorrt/download/10x)  、libtorch、[onnxruntime-win-x64-gpu-1.21.0](https://github.com/microsoft/onnxruntime/releases) 中的 `include` 、`lib` 、`share`目录复制到`ThirdParty`，某些库可能不存在`share`文件夹。

### 2. 获取模型文件

从[ZhouSiChuan/CosyVoiceCpp · Hugging Face](https://huggingface.co/ZhouSiChuan/CosyVoiceCpp)下载所有文件到项目的`model`目录

### 3. 编译工程

因为使用MSVC编译，所以本项目天生支持VS Studio编译，但笔者嫌VS Studio臃肿和懒惰没有尝试，下面讲解如何使用 VS code编译该工程。

安装基本的C++编译插件：[C/C++]、[C/C++ Extension Pack]、[CMake Tools]，安装完成后重启VS code

[Ctrl + shift + G]打开搜索选项栏，输入CMake，选择弹出的[CMake:Selectr a Kit/ CMake:选择工具包]，然后会有如下选项：

![image-20250802002741498](.\images\image-20250802002741498.png)

如果成功安装VS Studio，会自动扫描到VS 工具包，选择 x64 体系

**后续步骤需要电脑安装CMake， 本项目使用CMake 3.29.6**

然后再次[Ctrl + shift + G]打开搜索选项栏，输入CMake，选择[CMake:Configure/CMake:配置]，使用CMake生产构建系统所需配置文件。

在左侧扩展栏点击CMake，切换生成配置为`Release`，这非常关键！

![image-20250802003433032](.\images\image-20250802003433032.png)



最后再次[Ctrl + shift + G]打开搜索选项栏，输入CMake，选择[CMake:Build/CMake:生成] 生成可执行文件main.exe

### 4. 编辑指令进行语音合成

- 准备音色文件`spkInfo.jit`、`spkLines.json`

  由于笔者能力及精力有限，并未实现音色克隆的cpp化，仍需使用原python [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)项目进行音色克隆，我编写了一个简单脚本`spkdata.py`放置在项目根目录，用以方便地进行音色克隆：

  准备一段3s到10s的人声音频文件和其对应对白文本，并为其命名唯一地音色id。

  如：

  | 音色id    | 音频文件路径               | 对白                                     |
  | --------- | -------------------------- | ---------------------------------------- |
  | 原神-丽莎 | ./sample_dir/原神-丽莎.wav | 任务是做不完的，还不如配姐姐，聊会儿天。 |

  拷贝脚本`spkdata.py`到CosyVoice2项目根目录，确保CosyVoice2可正常运行。

  执行

  ```bash
  yourEnvDir/python.exe spkdata.py 原神-丽莎 ./sample_dir/原神-丽莎.wav
  ```

  ​		脚本执行完毕后，可在CosyVoice2项目根目录的`spkInfo`目录	找到`spkInfo.jit`文件，这就是语音合成所需的音色文件。将生成的`spkInfo.jit`文件拷贝至本项目的`model`文件夹。

  ​		同时在`model/spkLines.json`中添加对白，格式为

  ```json
  {
  	"原神-丽莎":"任务是做不完的，还不如配姐姐，聊会儿天。"
  }
  ```

- 指令说明

  ​	仓库根目录有文件`command.json`，用以对程序进行运行时的配置输入。其内容为：

  ```json
  {
  	"useCUDA": true,
  	"useTRT": true,
  	"seed": 300,
  	"spkId": "原神-迪卢克",
  	"ttsText": "今天天气不错，我们去散步吧。",
  	"instructText": "平和的语气"
  }
  ```

  | 字段         | 类型     | 释义                                                         |
  | ------------ | -------- | ------------------------------------------------------------ |
  | useCUDA      | bool     | 是否将模型加载至显存内(需英伟达显卡支持)                     |
  | useTRT       | bool     | 是否使用tensorRT模型加速推理(需英伟达显卡支持)               |
  | seed         | size_t   | torch推理种子值                                              |
  | spkId        | uint64_t | 进行语音合成时，使用的音色id，音色数据存储在`model\spkInfo.jit`中 |
  | ttsText      | string   | 用以语音合成的文本                                           |
  | instructText | string   | 情感控制文本，仅控制语音情感，不参与合成音频输出             |

  ​		本项目使用的模型文件可区分为三个模块`llm`、`flow`、`hift`。`llm`和`flow`都支持转换为tensorRT模型，`hift`因为算子兼容问题不支持，其中仅`flow`模块提供了tensorRT版本，它也是三个模块中推理开销最大的，且该tensorRT模型仅适用于40系显卡。开启`useCUDA`和`useTRT`会大幅提升推理速度，大概90%，但会消耗大量显存。

  > 因笔者能力及精力有限，将`llm`转换为tensorRT的合成效果极差，在`models`文件夹的`to_trt.md`中提供了各模型文件onnx转tensorRT的参考指令，说明了模型的动态轴名称及维度，爱好者可基于此及onnx开源特点，通过编辑节点或算子，获得更好的tensorRT推理效果。

- 开始合成

  在项目根目录打开终端，执行`chcp 65001`开启支持`UTF-8`

  启动程序：

  ```bash
  main.exe 1
  ```

  `1`表示开启日志

  ​		第一次运行程序时，会弹窗**缺失DLL**，可以直接把`cuDNN`、`TensorRT`、`libtorch`、`nvToolsExt`、`onnxRuntime`、`libsndfile`文件夹内提供的DLL全部拷贝到main.exe同级目录。其他缺失DLL可以打开文件搜索工具`everything`，从前文安装的依赖库的文件里找到缺失的DLL，并拷贝至main.exe同级文件夹。

  ​		如果控制台程序输出提示如下错误：
  
  ```
  INTEL MKL ERROR: �Ҳ���ָ����ģ�顣 mkl_vml_avx2.1.dll.
  Intel MKL FATAL ERROR: cannot load mkl_vml_avx2.1.dll or mkl_vml_def.1.dll.
  ```
  
  ​	可从[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice)的conda环境内找到`mkl_avx2.1.dll`、`mkl_vml_avx2.1.dll`，如：`C:\Users\31458\.conda\envs\cosyvoice\Library\bin\mkl_avx2.1.dll`
  
  ​	同样拷贝至main.exe 同级目录。
  
  当程序输出提示：
  
  ```
  准备好指令后, 按回车继续... 
  ```
  
  打开项目根目录的`command.json`文件，编辑合成令，你可以调整各字段值，以实现预期的合成效果。
  
  然后按下回车执行合成，合成完成后，程序会输出：
  
  ```
  In main, 👌👌👌音频保存成功: ../audio\20250802_214522_333.wav
  推理结束, 耗时: 6.666462 s
  准备好指令后, 按回车继续...
  ```
  
  可以看到生成的音频文件路径，以及可以再次编辑指令，然后按下回车执行下一次合成。

### 5. 其他

​	因转换tensorRT模型的损失，使用`useCUDA=false`、`useTRT=false`的合成效果最佳，但速度最慢。开启`useTRT`甚至会在某些音色下合成失败。

​	本项目在`spkInfo.jit`中准备了三个中文音色以供调试：`原神-丽莎`、 `原神-迪卢克`、`原神-诺艾尔`

​	可以使用[netron](https://netron.app/) 查看`spkInfo.jit`的具体结构
