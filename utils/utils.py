def load_classes(path):
    '''
    加载类别标签
    path:类别文件路径
    '''
    fp = open(path, 'r')
    names = fp.read().split('\n')[:]   # 读取每一行
    return names