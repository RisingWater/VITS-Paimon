原始项目：https://github.com/AlexandaJerry/vits-mandarin-biaobei

本项目为补档项目，原项目已经不能被看到了

![](IMG_9368.PNG)

进入VITS-Paimon目录,cd monotonic_align,python setup.py build_ext –inplace（两个短横，页面渲染的问题，显示一个短横，一个短横会引起错误），命令输出的最后一些内容是正在生成代码，已完成代码的生成。如果没有说完成代码的生成，只是列出了生成so文件的gcc命令，会发现monotonic_align对应的位置并没有那个文件，需要你手动创建必要的目录和执行这些命令。毕竟python在淘汰旧有的技术，这条命令所进行的工作比较老，新版本的库不完全支持它了。以后或许有新的方法。这样在导入monotonic_align这个包时，能执行from monotonic_align.core import xxxx语句。

将模型G_1434000.pth拷贝到项目目录中

设置语音文件名称和文本，开始合成语音

custom_synthesize.py合成一个文章

synthesize.py合成一个句子

只能合成中文语音
