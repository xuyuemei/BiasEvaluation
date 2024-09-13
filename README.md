1. 目的：对大语言模型生成的文本进行偏见评估，对比原维基百科文本的偏见评估，分析大语言模型的偏见情况、不同模型的偏见差异、以及与人类社会偏见的差异。

2. 项目示意图如下：
![image](https://github.com/user-attachments/assets/880023ef-03cf-4c91-bf87-fd6ce5d5e6d8)

3. 项目一共包括三个文件夹，分别为：Dataset、Evaluation_Metrics、以及Model。每个文件夹的说明如下 <br>
  Dataset：包含了本文所用的BOLD数据集，里面有提示和维基百科原文本 <br>
	Evaluation_Metrics: <br>
				&nbsp; &nbsp; &nbsp; 2.1 Distribution文件夹里是人口统计代表性指标（demographic representation）的代码（demographic representation.py）和使用的单词列表；<br>
   			&nbsp; &nbsp; &nbsp; 2.2 Gender_Polarity文件夹里是性别极性指标的代码，包括计算与性别直接相关的词（unigram_matching.py）和计算与性别间接相关的词(Gender-polarity.py)，copy版本是评估维基百科原文本； <br>
   			&nbsp; &nbsp; &nbsp; 2.3Sentiment文件夹里running文件夹是使用情感分类器（run_classification.py），VADER文件夹是使用VADER方法（VADER.py）； <br>
   			&nbsp; &nbsp; &nbsp; 2.4 Toxicity_model_Roberta文件夹是使用毒性分类器（Toxicity_Poberta.py）； 上述有些方法中运用了run.sh进行批量处理。<br>
  Model: <br>
				&nbsp; &nbsp; &nbsp; 包含了本文调用四个模型的代码（chatGLM.py，GLM-4.py，gpt3.5_2.0.py，t5-CPU.py）

4.数据集：<br>
<img width="452" alt="image" src="https://github.com/user-attachments/assets/c995cd8e-bedb-4f5b-988f-38e37240cbaf"> <br>

5. 评估指标的设计：<br>
<img width="1034" alt="image" src="https://github.com/user-attachments/assets/ca2e9db0-73f2-4533-8b82-addf72007d23"> <br>
<img width="1044" alt="image" src="https://github.com/user-attachments/assets/74bfc3c5-f9c2-4112-96b0-71224c5e6cfa">



5. 注：Sentiment/model_training下的model.pth文件下载链接：https://pan.baidu.com/s/1-r8cp8G9unkZ-SPRbASwOg  密码: s0sc。/Evaluation_Metrics/Toxicity_model_Roberta下的pytorch_model.bin下载链接：https://pan.baidu.com/s/1QPtMsfbmqYr6B31gWAwLpA  密码: 35wj。

 
