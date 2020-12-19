def parse_data_config(path):
    '''
    解析数据配置文件
    '''
    options = dict()    # 字典
    options['gpus'] = '0,1,2,3,4'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()   # 去掉空格键
        if line == '' or line.startswith('#'):   # 跳过空行和注释行
            continue
        key, value = line.split('=')    # 取出每行的键值对
        options[key.strip()] = value.strip()   # 存储在字典中
    return options

def parse_model_config(path):
    '''
    解析cfg网络配置文件，返回模型定义
    '''
    file = open(path, 'r')
    lines = file.read().split('\n')   # 逐行读取，返回列表
    lines = [x for x in lines if x and not x.startswith('#')]  # 去掉空行和注释行
    lines = [x.rstrip().lstrip() for x in lines]    # 去掉左右两边空格
    module_defs = []   # cfg文件内容以[]划分为各个块block，每个block对应一种操作例如卷积，将每个block读入存放到列表中
    for line in lines:
        if line.startswith('['):   # 遇到 [ 说明开始一个新的block
            module_defs.append({})     # 以字典的形式存储各个block的参数
            module_defs[-1]['type'] = line[1:-1].rstrip() # 存储block的type，例如卷积convolutional
            if module_defs[-1]['type'] == 'convolutional':     # 不是每个convolutional block都有batch_normalize参数，为了后面方便，给每个convolutional默认有batch_normalize参数且值为0
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')   # 以等号划分开      convolutional有batch_normalize的话会更新
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.lstrip()
    return module_defs