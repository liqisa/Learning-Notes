### 2019.6.24
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
