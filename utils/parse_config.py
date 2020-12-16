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
        line = line.strip()   # 去掉右边的空格键
        if line == '' or line.startswith('#'):   # 跳过空行和注释行
            continue
        key, value = line.split('=')    # 取出每行的键值对
        options[key.strip()] = value.strip()   # 存储在字典中
    return options
