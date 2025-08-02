# 将模型转换到TensorRT的指令

> 在 Windows CMD 中使用 `^` 作为行继续符
>
> 在 PowerShell 中使用反引号 `
>
> 只需要指定输入，不需要指定输出，输出会自动推导

### 1-1. llm_encoder 

✨✨

```bash
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\llm_encoder_export.onnx" --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\llm_encoder.trt" --minShapes=text:1x1,prompt_text:1x1,llm_prompt_speech_token:1x1 --optShapes=text:1x300,prompt_text:1x300,llm_prompt_speech_token:1x300 --maxShapes=text:1x3000,prompt_text:1x3000,llm_prompt_speech_token:1x3000 --best
```

### 1-2. llm_decoder_1_1

✨✨

```bash
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\llm_decoder_1_1_export.onnx" 
			 --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\llm_decoder_1_1.trt" 
			 --minShapes=lm_input:1x1x896
			 --optShapes=lm_input:1x300x896
			 --maxShapes=lm_input:1x3000x896
			 --best
```

### 1-2. llm_decoder_1_2

✨✨

```bash
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\llm_decoder_1_2_export.onnx" 
			 --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\llm_decoder_1_2.trt" 
			 --minShapes=cache_input_0:1x2x1x64,cache_input_1:1x2x1x64,cache_input_2:1x2x1x64,cache_input_3:1x2x1x64,cache_input_4:1x2x1x64,cache_input_5:1x2x1x64,cache_input_6:1x2x1x64,cache_input_7:1x2x1x64,cache_input_8:1x2x1x64,cache_input_9:1x2x1x64,cache_input_10:1x2x1x64,cache_input_11:1x2x1x64,cache_input_12:1x2x1x64,cache_input_13:1x2x1x64,cache_input_14:1x2x1x64,cache_input_15:1x2x1x64,cache_input_16:1x2x1x64,cache_input_17:1x2x1x64,cache_input_18:1x2x1x64,cache_input_19:1x2x1x64,cache_input_20:1x2x1x64,cache_input_21:1x2x1x64,cache_input_22:1x2x1x64,cache_input_23:1x2x1x64,cache_input_24:1x2x1x64,cache_input_25:1x2x1x64,cache_input_26:1x2x1x64,cache_input_27:1x2x1x64,cache_input_28:1x2x1x64,cache_input_29:1x2x1x64,cache_input_30:1x2x1x64,cache_input_31:1x2x1x64,cache_input_32:1x2x1x64,cache_input_33:1x2x1x64,cache_input_34:1x2x1x64,cache_input_35:1x2x1x64,cache_input_36:1x2x1x64,cache_input_37:1x2x1x64,cache_input_38:1x2x1x64,cache_input_39:1x2x1x64,cache_input_40:1x2x1x64,cache_input_41:1x2x1x64,cache_input_42:1x2x1x64,cache_input_43:1x2x1x64,cache_input_44:1x2x1x64,cache_input_45:1x2x1x64,cache_input_46:1x2x1x64,cache_input_47:1x2x1x64
			  --optShapes=cache_input_0:1x2x500x64,cache_input_1:1x2x500x64,cache_input_2:1x2x500x64,cache_input_3:1x2x500x64,cache_input_4:1x2x500x64,cache_input_5:1x2x500x64,cache_input_6:1x2x500x64,cache_input_7:1x2x500x64,cache_input_8:1x2x500x64,cache_input_9:1x2x500x64,cache_input_10:1x2x500x64,cache_input_11:1x2x500x64,cache_input_12:1x2x500x64,cache_input_13:1x2x500x64,cache_input_14:1x2x500x64,cache_input_15:1x2x500x64,cache_input_16:1x2x500x64,cache_input_17:1x2x500x64,cache_input_18:1x2x500x64,cache_input_19:1x2x500x64,cache_input_20:1x2x500x64,cache_input_21:1x2x500x64,cache_input_22:1x2x500x64,cache_input_23:1x2x500x64,cache_input_24:1x2x500x64,cache_input_25:1x2x500x64,cache_input_26:1x2x500x64,cache_input_27:1x2x500x64,cache_input_28:1x2x500x64,cache_input_29:1x2x500x64,cache_input_30:1x2x500x64,cache_input_31:1x2x500x64,cache_input_32:1x2x500x64,cache_input_33:1x2x500x64,cache_input_34:1x2x500x64,cache_input_35:1x2x500x64,cache_input_36:1x2x500x64,cache_input_37:1x2x500x64,cache_input_38:1x2x500x64,cache_input_39:1x2x500x64,cache_input_40:1x2x500x64,cache_input_41:1x2x500x64,cache_input_42:1x2x500x64,cache_input_43:1x2x500x64,cache_input_44:1x2x500x64,cache_input_45:1x2x500x64,cache_input_46:1x2x500x64,cache_input_47:1x2x500x64
			 --maxShapes=cache_input_0:1x2x4000x64,cache_input_1:1x2x4000x64,cache_input_2:1x2x4000x64,cache_input_3:1x2x4000x64,cache_input_4:1x2x4000x64,cache_input_5:1x2x4000x64,cache_input_6:1x2x4000x64,cache_input_7:1x2x4000x64,cache_input_8:1x2x4000x64,cache_input_9:1x2x4000x64,cache_input_10:1x2x4000x64,cache_input_11:1x2x4000x64,cache_input_12:1x2x4000x64,cache_input_13:1x2x4000x64,cache_input_14:1x2x4000x64,cache_input_15:1x2x4000x64,cache_input_16:1x2x4000x64,cache_input_17:1x2x4000x64,cache_input_18:1x2x4000x64,cache_input_19:1x2x4000x64,cache_input_20:1x2x4000x64,cache_input_21:1x2x4000x64,cache_input_22:1x2x4000x64,cache_input_23:1x2x4000x64,cache_input_24:1x2x4000x64,cache_input_25:1x2x4000x64,cache_input_26:1x2x4000x64,cache_input_27:1x2x4000x64,cache_input_28:1x2x4000x64,cache_input_29:1x2x4000x64,cache_input_30:1x2x4000x64,cache_input_31:1x2x4000x64,cache_input_32:1x2x4000x64,cache_input_33:1x2x4000x64,cache_input_34:1x2x4000x64,cache_input_35:1x2x4000x64,cache_input_36:1x2x4000x64,cache_input_37:1x2x4000x64,cache_input_38:1x2x4000x64,cache_input_39:1x2x4000x64,cache_input_40:1x2x4000x64,cache_input_41:1x2x4000x64,cache_input_42:1x2x4000x64,cache_input_43:1x2x4000x64,cache_input_44:1x2x4000x64,cache_input_45:1x2x4000x64,cache_input_46:1x2x4000x64,cache_input_47:1x2x4000x64
			 --best
```





### 1-3. llm_decoder_2

✨✨

```bash
固定尺寸, 不用设置形状
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\llm_decoder_2_export.onnx" 
			 --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\llm_decoder_2.trt"
			 --best
```



### 2-1.  flow_1

✨✨

```bash
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\flow_1_export.onnx" 
			 --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\flow_1.trt"
			 --minShapes=token:1x1,mask:1x1x1
			 --optShapes=token:1x300,mask:1x300x1
			 --maxShapes=token:1x3000,mask:1x1000x1
			 --best
```

### 2-2. flow_2_1

✨✨

```bash
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\flow_2_1_export.onnx" 
			 --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\flow_2_1.trt"
			 --best
```

### 2-3. flow_2_2

✨✨

```bash
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\flow_2_2_export.onnx" 
			 --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\flow_2_2.trt"
			 --minShapes=x:2x80x1,mask:2x1x1,mu:2x80x1,cond:2x80x1
			 --optShapes=x:2x80x300,mask:2x1x300,mu:2x80x300,cond:2x80x300
			 --maxShapes=x:2x80x2000,mask:2x1x2000,mu:2x80x2000,cond:2x80x2000
			 --best
```

### 3-1.  hift_1

```bash
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\hift_1_export.onnx" 
			 --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\hift_1.trt"
			 --minShapes=speech_feat:1x80x1
			 --optShapes=speech_feat:1x80x300
			 --maxShapes=speech_feat:1x80x3000
			 --best
```

### 3-3. hift_2

```bash
.\trtexec.exe --onnx="F:\Zlab\C++\002-C++\116-libTorch\model\hift_2_export.onnx" 
			 --saveEngine="F:\Zlab\C++\002-C++\116-libTorch\model\hift_2.trt"
			 --minShapes=speech_feat:1x80x1,s_stft_real:1x9x1,s_stft_imag:1x9x1
			 --optShapes=speech_feat:1x80x300,s_stft_real:1x9x30000,s_stft_imag:1x9x30000
			 --maxShapes=speech_feat:1x80x3000,s_stft_real:1x9x300000,s_stft_imag:1x9x300000
			 --best
```

