### pytorch
1. 2019.6.24
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
```nn.Sequential()```的输入可以是(conv1,conv2,covn3)类似的顺序模块，也可以是orderDict，也可以是List,但是必须加 * 来引用
