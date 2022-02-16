# DEoR tool
**你好！**

这些代码主要是出于对宇宙再电离时期全天总功率探测实验进行模拟观测以及测试信号提取方法而编写的，其中暂时包括LGSM和DEORtool两个部分，它们是基于PyGSM2008模型开发的（见https://github.com/telegraphic/pygdsm ），如果你想要使用这些代码，那么你首先要在前面的链接中安装PyGSM模型，然后将这些文件放到PyGSM模型下的pygdsm目录。

LGSM目前基本上是基于pygsm2008模型的，但是在其中增加了同时生成多个频率的观测天图功能，以及一些可视化方法。

DEORtool中目前有射电银河前景幂律谱的生成与可视化，银河射电前景的全天平均温度在一天中的变化，以及一些常用的再电离信号模型（Bowman2008，EDGES2018等等）

未来将会继续加入更多功能：生成大量的模拟观测数据，加入电离层影响，加入随机噪声与热噪声，基于FEKO软件的总功率实验所使用的天线beam模拟等等

暂时面临的问题：在计算大量数据时（如生成的频率分辨率小于100KHz，时间为1440min）速度不够快

**问：为什么不使用GSM2016模型？**

**答：GSM2016模型在大约85MHz处有一微小跃变，似乎是其PCA数据本身的问题，暂时无法排除**
