import utils
from models import SynthesizerTrn
from text import text_to_sequence
import commons
import torch
from text.symbols import symbols
import sys
import time
import os
import subprocess

def get_text(text,hps):
	text_norm=text_to_sequence(text,hps.data.text_cleaners)
	if hps.data.add_blank:
		text_norm=commons.intersperse(text_norm,0)
	text_norm=torch.LongTensor(text_norm)
	return text_norm

def format_duration(duration):
	return f'{duration//3600}小时{duration%3600//60}分钟{duration%3600%60}秒'

hps=utils.get_hparams_from_file('configs/biaobei_base.json')

net_g=SynthesizerTrn(len(symbols),hps.data.filter_length//2+1,hps.train.segment_size//hps.data.hop_length,**hps.model).cuda()
_=net_g.eval()

_=utils.load_checkpoint('G_1434000.pth',net_g,None)
import soundfile as sf
text='''
3.6  测试用例 
3.6   测试用例  
在测试时，用例也有一定的用途。这里，需要执行两种类型的测试：验证（Verification）和确
认（Validation）。前者进一步证实系统是否正确地开发或者是否是依照制定的规格说明进行开发的；
后者用于确保正在开发的系统确实是客户或最终用户真的需要的。  UML 工具箱  
Java 爱好者     http://www.javafan.net 制作  
确认测试是在开发过程的前段进行的。只要完成了一个用例模型的建模（或者可能是在该用例
模型的开发过程中），就提交该模型，并与客户及最终用户一起来进行讨论。客户和最终用户必须确
认该模型的正确性，以及该模型是否完全满足他们对系统的期望。其中，特别需要确认的是系统为
他们提供的那些功能。为做到这一步，开发人员必须确保客户真正地理解了该模型和它的意图，以
排除那些不可接受的内容。在此过程中，毫无疑问，将会出现更多的问题和思想，在客户和最终用
户最后确认之前需要将它们添加到用例模型中。确认测试也可以在系统测试阶段进行，但是，这时
出现的问题是：如果该系统不满足用户提出的需求，那么整个项目就可能不得不重新开始。  
系统验证测试的目的是测试系统是否是依照规格说明所规定的那样运行的。因此，只有当系统
的某些部分能够运行时，才能进行验证测试。这时，就有测试系统的行为是否是按照用户指定的那
样进行的，并测试在用例模型中所描述的那些用例的执行情况，以及它们是否是按照用例描述中所
描述的那样来运转的可能性。  
3.6.1   排练用例  
在实现用例定义和用例测试的过程中，有一种非常好的技术：排练用例（Walking the use case）。
应用这项技术时，不同的人按照分配的角色扮演一个指定用例中的那些参与者或系统本身。排练过
程由扮演参与者角色的人启动，启动方法是：他说：参与者使用本系统做这个动作。就会导致系统
执行由此动作启动的一个特定用例。然后，扮演系统角色的那个人说：当该用例执行时他（代表本
系统）做什么事情。在此过程中，开发人员并不参与角色的扮演，而是在一旁作记录，并试着发现
由这些扮演者所描述的那些用例中的不足之处。一般来说，通过这种方法，开发人员可以发现有些
可选的路径还没有被描述，以及有些动作的描述还不够详细。  
在应用排练用例这种技术时，如果那些角色扮演者对系统的用法越熟悉，那么对用例的测试效
果就越好。所以让那些角色扮演者相互交换角色，可以得到对系统的不同的诠释和观点，为建模人
员提供更多的信息，从而使用例的描述更加明确，同时也为建模人员指出他们遗漏的事情。当所有
参与者的角色都被扮演了，并且所有的用例都以这种方式被执行了，那么，该用例模型的一个完整UML 工具箱  
Java 爱好者     http://www.javafan.net 制作  
测试过程也就完成了。  
 
3.7  实现用例 
3.7   实现用例  
用例是系统的功能描述，它并不关心这些功能是如何实现的。这就意味着这些用例将在系统中
被实现，也就是将那些职责（负责完成用例描述中所描述的动作）分配给系统中的多个对象（相互
合作，负责实现描述的功能）。  
UML 实现用例的原则有以下几点。  
n         在协作中实现用例：协作可以利用各个类/对象和它们之间的关系（称之为协作上下文
（Context）），以及为了达到期望的功能在这些类/对象之间所进行的交互（称之为协作交互
（Interaction）），来显示出用例的一个与实现相关的内部解决方案。在 UML 中，表示协作的符号
是一个包含了该协作名称的椭圆。 
n         在 UML 中，一个协作由多个图表示，这些图显示了协作的参与者之间的上下文和交互：
协作的参与者包括许多类（并且，一个协作实例中的参与者是对象）。那些表示协作的图是协作图、
顺序图和活动图。在实际工作中，我们用哪种图去描绘协作的完整画面取决于实际的用例。在某些
情况下，一个协作图就足够了，但在另一些情况下，可能需要多个不同的图结合起来才能够描述整
个协作。 
n         场景是用例或协作的一个实例：场景是系统的一个特定执行路径（一个特定的事件流），
代表了用例（系统的一种用法）的一个特殊例示。当场景被看做是一个用例的例示时，只需要描述
系统对参与者的外部行为；当场景被看做是一个协作的例示时，需要描述该协作涉及的那些类、类
的操作，以及这些类之间的通信的内部实现。 UML 工具箱  
Java 爱好者     http://www.javafan.net 制作  
实现一个用例的任务就是将该用例描述中那些不同步骤和动作（在用例描述文本中或活动图中描述
的内容）转换为各个类、类的操作以及这些类之间的关系。它是这样描述的：将该用例中的每一步职责
分配给参与此协作（实现了该用例）的所有类。在此阶段，我们可以找到一个解决方案，该方案给出了
用例已经指定的外部行为，并且是利用系统内部的一个协作来描述的，如图 3.9 所示。  
用例描述中的每一步都被转换为多个类（这些类参与了实现该用例的协作）的操作。也就是说，
用例中描述的一步动作将被转换为这些类的多个操作。这时，用例中的动作和对象（指那些参加协
作的类的对象）交互中的一个操作之间，不太可能是一对一的关系。同时，读者也要注意：一个类
可以参与多个用例，所以类的全部职责是它在所有用例中扮演的所有角色的集合。  
为了显示用例和它的实现之间的关系，我们可以用精化关系来表示，在 UML 中，该关系用一
条带有箭头的虚线表示，如图 3.9 所示；另外，也可以通过其他方式，例如在建模工具中用一个不
可见的超链接来表示。这种工具中的超链接使得用户有可能从观看用例图中的一个用例切换到观看
实现该用例的真实的协作。超链接也用于从一个用例切换到一个描述了该用例的特定执行（该用例
的一个例示）的场景（一般来说，它是任意的动态模型：活动图、顺序图，或者是协作图）。  
 
图 3.9  一个用例在协作中被实现，并且多个类参与了此协作。为了实现一个用例，用例内的
每一步职责必须根据关系和操作被转换为各个类之间的协作 
将职责成功地分配给各个类是一项要求实施者具有一定经验的任务。在任何情况下，一项工作
一旦涉及到面向对象，那么它就是一项需要多次反复进行的工作。开发人员尝试不同的可能性，逐UML 工具箱  
Java 爱好者     http://www.javafan.net 制作  
渐地改进解决方案，直到完成了能够执行指定功能的模型，并且该模型足够灵活，允许将来对其进
行修改（本书第 4 章将讨论模型质量方面的内容）。  
关于实现用例，Jacobson 使用了一种定义了三种构造型对象类型（如：类）的方法，这三种构
造型对象类型是：边界对象（以前称之为接口对象）、控制对象和实体对象。对每一个用例来说，这
些对象都用于描述实现该用例的协作。这三种构造型各自的职责如下所述。  
n         边界对象（Boundary Object）：这种类型的对象紧邻着系统的边界（但仍然属于系统内
部）。它们与系统外部的参与者打交道，后者通过这些边界对象与系统内部的其他类型对象进行消
息交互。 
n         控制对象（Control Object）：这种类型的对象控制了一组对象之间的交互。这样的对象
可以是一个完整用例的“控制者”，也可能实现了几个用例的一个公共活动流程。通常，这样的对
象仅仅存在于用例的执行过程中。 
n         实体对象（Entity Object）：这种类型的对象代表系统处理领域内的问题域实体。一般来
说，这种类型的对象是被动的，它自己不启动交互。在信息系统中，实体对象通常是永久的，并且
存储在数据库中。一般来说，实体对象参与多个用例。 
这些类的构造型都有它们自己的图标（与 Objectory 方法中使用的符号类似），当绘制那种描述
了一个协作的图时，或者是在一个类图中，都可以使用这些图标。在定义了不同对象类型并指定了
它们之间的协作之后，就可以进行一项特殊的活动，目的是寻找它们之间存在的相同点，这样，有
些类就可以在多个用例中使用。以这种方式应用用例也可以作为系统分析和设计的基础。这样的开
发过程就是 Jacobson 所命名的用例驱动。  
当从用例将职责分配到各个类时，不同的方法有不同的建议。一些方法建议应该进行一次问题
域分析，显示问题域的所有类及它们之间的关系。然后，开发人员再将用例中的职责分配给分析模
型中的各个类，在这一过程中，有时还需要进行一些修改或者加入一些新的类。另一些方法则建议
将用例作为发现类的基础，在分配职责过程中逐渐建立问题域的分析模型。  UML 工具箱  
Java 爱好者     http://www.javafan.net 制作  
这里，再次着重强调：这项工作是迭代式的。当职责被分配给各个类时，开发人员可能会发现
类图中存在错误和遗漏，需要对类图进行修改。当然，为了支持用例，也会增加一些新的类。在某
些情况下，甚至可能有必要对用例图进行修改，因为随着对系统的更深入理解，开发人员可能会认
识到对某个用例的描述是不正确的。用例帮助我们把注意力集中在系统的功能上，这样，系统的功
能就可以被正确地描述，系统也就可以正确地实现。没有用例的那几种面向对象方法存在的一个问
题是：它们专注于类和对象的静态结构（有时称之为概念建模），而忽略正在开发的系统的功能和动
态方面的问题。  
 
3.8  本章小结 
3.8   本章小结  
用例建模是一项用于描述系统功能需求的技术。用例是根据外部参与者、用例和被建模的系统
来描述的。其中，参与者代表一个外部实体（如用户、硬件或其他系统）与系统交互时所扮演的角
色。参与者启动用例，并与用例进行通信，这里，用例就是系统执行的一组活动序列。一个用例必
须向参与者交付一个具体的值，并且通常用例是通过文本文档进行描述的。参与者和用例是多个类。
参与者通过关联与一个或多个用例相连，并且参与者和用例都可以具有泛化关系，这种关系描述了
由一个或多个特化的子类所继承的超类中的公共行为。用例模型是在一个或多个 UML 用例图中描
述的。  
用例是在协作中实现的，协作是上下文（显示了各个类/对象以及它们之间的关系）和交互（显
示了各个类/对象如何交互，以完成一个特定的功能）的描述。协作是用活动图、协作图和交互图（将
在本书第 5 章进行描述）描述的。当实现一个用例时，该用例中每一步动作的职责必须分配给参与
该协作的那些类，一般来说，实现这种分配的方式是：指定这些类的操作，同时指定这些类之间的
交互方式。场景是用例或协作的一个实例，显示了系统的一个特定执行路径。当场景被看做是用例UML 工具箱  
Java 爱好者     http://www.javafan.net 制作  
的实例时，仅仅需要描述用例和外部参与者之间的交互；而当场景被看做是协作的实例时，需要描
述系统内的类/对象之间的交互，后者实现了系统，如图 3.10 所示。  
 
图 3.10  用例、协作及场景之间的关系 
'''
final_filename='3.6测试用例_3.7实现用例_3.8本章小结'

length_scale=1.0


if os.path.exists('synthesize_files.txt'):
	os.remove('synthesize_files.txt')

text=text.replace('\n','').replace('  ',' ').replace('/','-')

old_sentence=''
sentences=text.split('。')
count=0
for i,sentence in enumerate(sentences):
	st=time.time()
	sentence=old_sentence+sentence
	if len(sentence)==0:
		continue
	if len(sentence)<10:
		old_sentence=sentence
		if i<len(sentences)-1:
			continue
	count+=len(sentence)
	print(f'processing {sentence}')
	filename=f'{sentence[:10]}_{time.strftime("%y%m%d%H%M%S")}'.replace(' ','_').replace('(','_').replace(')','_')
	audio_path=f'synthesizes/{filename}.wav'
	stn_tst=get_text(sentence,hps)
	with torch.no_grad():
		x_tst=stn_tst.cuda().unsqueeze(0)
		x_tst_lengths=torch.LongTensor([stn_tst.size(0)]).cuda()
		audio=net_g.infer(x_tst,x_tst_lengths,noise_scale=.667,noise_scale_w=0.8,length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
	sf.write(audio_path,audio,samplerate=hps.data.sampling_rate)
	with open('synthesize_files.txt','a',encoding='utf8') as f:
		f.write(f'{filename}.wav\n')
	duration=time.time()-st
	print(f'本次合成花费{duration}秒，还需要合成{len(sentences)-i-1}条句子，花费{format_duration(duration/len(sentence)*(len(text)-count))}，synthesizing {filename}.wav')
	old_sentence=''

with open('synthesize_files.txt','r',encoding='utf8') as f:
	lines=f.read().strip().split('\n')

os.chdir('synthesizes')
cmd_str='ffmpeg '
tail='-filter_complex "'
tail_tail='[0:a]'
for i,line in enumerate(lines):
	cmd_str+='-i '+line+' '
	if i>0:
		tail+=f'[{i}:a]adelay=1000[a{i}];'
		tail_tail+=f'[a{i}]'
cmd_str+=tail+tail_tail+f'concat=n={len(lines)}'+f':v=0:a=1" {final_filename}.wav'
print(cmd_str)
subprocess.run(cmd_str.encode('utf8').decode('utf8'),shell=True)
