# DEoR tool
**你好！**

这些代码主要是出于对宇宙再电离时期全天总功率探测实验进行模拟观测以及测试信号提取方法而编写的，其中暂时包括LGSM和DEORtool两个部分，它们是基于PyGSM2008模型开发的（见https://github.com/telegraphic/pygdsm ），如果你想要使用这些代码，那么你首先要在前面的链接中安装PyGSM模型，然后将这些文件放到PyGSM模型下的pygdsm目录。

LGSM目前基本上是基于pygsm2008模型的，但是在其中增加了同时生成多个频率的观测天图功能，以及一些可视化方法。

DEORtool中目前有射电银河前景幂律谱的生成与可视化，银河射电前景的全天平均温度在一天中的变化，以及一些常用的再电离信号模型（Bowman2008，EDGES2018等等）

未来将会继续加入更多功能：生成大量的模拟观测数据，加入电离层影响，加入随机噪声与热噪声，基于FEKO软件的总功率实验所使用的天线beam模拟等等

**2.17更新，加入了降低已生成天图分辨率的功能，该功能可将生成的314万像素的天图降低到需要的精度，显著提高大量计算时的运算速度**

**2.24更新，加入了从FEKO导出的ffe远场文件中读取仿真天线的方向图数据，然后进行插值得到频率下的天线beam，结果可以直接与mask之后的observed_sky进行相乘，得到相应的模拟观测数据，需要注意的是，ffe文件要经过预处理为测试文件类似的csv文档后才能使用本程序**

**3.3更新，加入了一个新的天线模型，它具有更高的分辨率，插值之后的效果更好**

**问：为什么不使用GSM2016模型？**

**答：GSM2016模型在大约85MHz处有一微小跃变，似乎是其PCA数据本身的问题，暂时无法排除**

**一些问题：三维插值时无法使用cubic方法，使用nearest和linear插值方法的效果不是特别理想，可能提高FEKO模拟精度后再计算更好**
