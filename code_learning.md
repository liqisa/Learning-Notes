### 2019.06.24
1. pytorch
```python
def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)
```
Ref: https://blog.csdn.net/u013548568/article/details/80294708  
```nn.Sequential()``` 的输入可以是(conv1,conv2,covn3)类似的顺序模块，也可以是orderDict，也可以是List,但是必须加 * 来引用

2. conda env配置
安装cv2 PIL(pillow) matplotlib时，最后安装 cv2,否则会导致cv2不可用  
```import cv2```时出现 ```ImportError: numpy.core.multiarray failed to import```的问题   
torchvision0.3.0 搭配 pytorch1.0.1会出现问题，需降级到0.2.1

### 2019.07.12
1. pytorch
```python
def split_data(imgs, targets, split_imgs=True):
    """ Liqi 2019-7-9
    :param imgs: input imgs in this batch .
    :param targets: input targets in this batch.
    :param split_imgs: if need split imgs into bk and jh task.
    :return:
    """
    num = len(targets)
    if num != imgs.size(0):
        print('Error: Target and Imgs sizes not match')
        sys.exit()
    bk_index, jh_index = [],[]
    targets_bk, targets_jh = [],[]
    imgs_bk, imgs_jh = torch.tensor(()), torch.tensor(())
    for index in range(num):
        flag = targets[index][:1,-2:].data
        #if flag
        #print(type(flag))
        if flag.equal(torch.Tensor([[1.,-1.]])):
            bk_index.append(index)
            targets_bk.append(targets[index][:,:-2])
            if split_imgs:
                imgs_bk = torch.cat((imgs_bk, imgs[index].unsqueeze(0)))
        if flag.equal(torch.Tensor([[-1., 1.]])):
            jh_index.append(index)
            targets_jh.append(targets[index][:, :-2])
            if split_imgs:
                imgs_jh = torch.cat((imgs_jh, imgs[index].unsqueeze(0)))
    if split_imgs:
        return imgs_bk,imgs_jh,targets_bk,targets_jh
    else:
        return targets_bk,targets_jh
```
- tensor.unsequeeze(x) 可用于张量扩充维度，增加一个新的大小为1的维度，在torch.cat()前，x表示要扩充的维度位置  
> size为[3,3]的张量经过unsequeeze(0)后变成[1,3,3]  
同理：sequeeze()为减少维度  
Ref:   

- 关于nn.ModuleList() 模型列表
> ModuleList是Module的子类，可被自动识别为Module()
- torch的一些张量变换操作：
    - permute()
    - view()
    - stack() 与numpy的stack vstack hstack 相同用法

2. python

- enumerate()操作，常用于从dataLoader中获取数据
```python
    for iteration, (imgs, targets, _) in enumerate(train_loader):
        # Imgs and Targets in a batch
        t0 = time.time()
        lr = adjust_learning_rate(optimizer, epoch, epoch_step, gamma,
                                  epoch_size, iteration)
        imgs = imgs.cuda()
        targets = [anno.cuda() for anno in targets]

```
- zip()操作
```python
a = [1,2,3]
b = [4,5,6]
c = zip(a,b)
print(c) # 不可直接输出
>>> <zip object at 0x7f068c5ccd48>
print(list(c)) # 转成list输出
>>> [(1,4),(2,5),(3,6)]
print(*c) # 直接引用 
>>> (1,4),(2,5),(3,6)
```

### 2019.07.15

#### linux **tar** 指令使用指南  

1. tar 指的是对一个或者多个文件或者文件夹进行打包，合并成一个文件，而 gzip bzip2等指令是对一个大的文件进行压缩，变成一个更小的文件【压缩操作大多只能对一个文件进行，所以一般先tar打包后再压缩】
2. **tar 语法**  

语法：tar [主选项+辅选项] 文件或目录

使用该命令时，主选项必须有，它告诉tar要做什么事情，辅选项是辅助使用的，可以选用  
主选项：【一条命令以下5个参数只能有一个  
```-c```: --create 新建一个压缩文档，即打包  
```-x```: --extract,--get解压文件  
```-t```: --list,查看压缩文档里的所有内容  
```-r```:--append 向压缩文档里追加文件
```-u```:--update 更新原压缩包中的文件  
辅助选项：  
```-z```:是否同时具有gzip的属性？即是否需要用gzip压缩或解压？一般格式为xxx.tar.gz或xx.tgz  
```-j```：是否同时具有bzip2的属性？即是否需要用bzip2压缩或解压？一般格式为xx.tar.bz2  
```-v```:显示操作过程！这个参数很常用  
```-f```：使用文档名，注意，在f之后要立即接文档名，不要再加其他参数！  
```-C```:切换到指定目录  
```--exclude FILE```:在压缩过程中，不要将FILE打包

3. **例子**
```shell
$ tar -cvf img.tar img1 img2
# 将img1和img2两个文件夹打包成img.tar，仅打包不压缩
img1/
img1/102.png
img1/101.png
img1/100.png
img2/
img2/105.png
img2/104.png
img2/103.png
$ ls
img1  img2  img.tar
```

```shell
# 将img1和img2两个文件夹打包成img.tar.gz，打包后，以gzip压缩
$ tar -zcvf img.tar.gz img1 img2
img1/
img1/102.png
img1/101.png
img1/100.png
img2/
img2/105.png
img2/104.png
img2/103.png
$ ls
img1  img2  img.tar  img.tar.gz
```

4. 更多内容，参考 https://www.cnblogs.com/starof/p/4229017.html

### 2019.07.17
- Python中的 ```collections.defaultdict()```

1. 例子1：统计一个list中各个元素出现的次数
```python
words = ['hello', 'world', 'nice', 'world']
counter = dict()
for kw in words:
    if kw in words:
        counter[kw] += 1
    else:
        counter[kw] = 0
```
2. 使用setdefault()方法设置默认值
```python
words = ['hello', 'world', 'nice', 'world']
counter = dict()
for kw in words:
    counter.setdefault(kw, 0)
    counter[kw] += 1
```
> setdefault()，需提供两个参数，第一个参数是键值，第二个参数是默认值，每次调用都有一个返回值，如果字典中不存在该键则返回默认值，如果存在该键则返回该值，利用返回值可再次修改代码。
```python
words = ['hello', 'world', 'nice', 'world']
counter = dict()
for kw in words:
    counter[kw] = counter.setdefault(kw, 0) + 1
```

3. 使用 ```collections.defaultdict()```
> 一种特殊类型的字典本身就保存了默认值defaultdict()，defaultdict类的初始化函数接受一个类型作为参数，当所访问的键不存在的时候，可以实例化一个值作为默认值。
```python
from collections import defaultdict
dd = defaultdict(list)
defaultdict(<type 'list'>, {})
#给它赋值，同时也是读取
dd['h'h]
defaultdict(<type 'list'>, {'hh': []})
dd['hh'].append('haha')
defaultdict(<type 'list'>, {'hh': ['haha']})
```
> 该类除了接受类型名称作为初始化函数的参数之外，还可以使用任何不带参数的可调用函数，到时该函数的返回结果作为默认值，这样使得默认值的取值更加灵活。

```shell
>>> from collections import defaultdict
>>> def zero():
...     return 0
...
>>> dd = defaultdict(zero)
>>> dd
defaultdict(<function zero at 0xb7ed2684>, {})
>>> dd['foo']
0
>>> dd
defaultdict(<function zero at 0xb7ed2684>, {'foo': 0})
```

```python
from collections import defaultdict
words = ['hello', 'world', 'nice', 'world']
#使用lambda来定义简单的函数
counter = defaultdict(lambda: 0) 
for kw in words:
    counter[kw] += 1
```
4. Ref: https://www.jianshu.com/p/26df28b3bfc8

###2019.09.09
#### python list中元素的删除
1. remove()
> 描述：remove()函数用于移除列表中某个值的第一个匹配项  
语法：list.remove(obj)     obj---列表中要移除的对象  
返回值：没有返回值，但是会移除列表中的某个值的第一个匹配项  
总结：remove()删除单个元素，删除首个符合条件的元素，按值删除，返回为空  

```shell
>>> l = [1,3,4,5]
>>> print(l.remove(4))
>>> print(l)
None
[1, 3, 5]

>>> l = [1,3,4,4,4,5]
>>> print(l.remove(4))
>>> print(l)
None
[1, 3, 4, 4, 5]    #只删除首个符合条件的元素
```
2. del
> 根据索引位置来删除单个值或者指定范围的值。  
删除变量而不是数据，解除某个变量对数据的引用   
```python
>>> l = [1,3,4,4,4,5]
>>> del l[0]
>>> print(l)
[3, 4, 4, 4, 5]

>>> l = [1,3,4,4,4,5]
>>> del l[0:3]
>>> print(l)
[4,4,5]

>>> l = [1,3,4,4,4,5]
>>> del l   #删除后，找不到对象
>>> l
NameError: name 'l' is not defined


>>> a=1       # 对象 1 被 变量a引用，对象1的引用计数器为1
>>> b=a       # 对象1 被变量b引用，对象1的引用计数器加1
>>> c=a       #1对象1 被变量c引用，对象1的引用计数器加1
>>> del a     #删除变量a，解除a对1的引用
>>> del b     #删除变量b，解除b对1的引用
print(c)  #最终变量c仍然引用1
```
3. pop()
> 描述：pop()函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值  
语法：list.pop([index=-1])   index---可选参数，要移除列表元素的索引值，不能超过列表的总长度，默认index=-1，删除最后一个列表值。  
返回值：返回从列表中移除的元素对象  
总结：pop()删除索引位置元素，无参情况下默认删除最后一个元素，返回删除的元素值  
```python
>>> l = [1,3,4,4,4,5]
>>> print(l.pop(0))
>>> print(l)
 
1
[3, 4, 4, 4, 5]
```