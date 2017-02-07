# Caicloud 大数据平台 python 包

## 关于文件

setup.py 是 python 的打包文件。

README.rst 是关于 Caicloud 大数据平台 python 包的详细说明信息。

caicloud 目录则是要打包的 python package。

## 本地开发

切换到当前目录（python）下，然后运行下面命令进行安装。

```shell
$ pip install -e .
```

这个命令只是在 /Library/Python/2.7/site-packages 目录下建一个包链接，链接到当前目录。
我们在本地修改的内容立即生效。

## 打包

运行下面命令打包。

```shell
$ python setup.py sdist
```

该命令生成两个目录：
- caicloud.tensorflow.egg-info：python 包的 meta 信息。
- dist：python 包目录，该目录下会生成 caicloud.tensorflow-x.x.x.tar.gz 的 python 包文件，其中 x.x.x 就是版本号。

## 验证 tar.gz 包正确性

运行下面命令来通过 tar.gz 包来进行安装，

```shell
$ pip install ./dist/caicloud.tensorflow-x.x.x.tar.gz
```

然后进入 /Library/Python/2.7/site-packages 目录查看是否有 caicloud 的目录，并且该目录下的文件是否完整。

## 发布

发布之前要先在 ~/.pypirc 文件中设置 PyPI 的帐号信息，

```
[distutils]
index-servers=pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = caicloud
password = caicloud2015ABC
```

使用 twine 进行发布。

```shell
$ twine upload dist/caicloud.tensorflow-x.x.x.tar.gz
```

发布完之后，便可以 https://pypi.python.org/pypi 中查看我们发布的包。
