import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)   # 创建一个日志记录，保存训练数据，log_dir是日志保存路径

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
    
    def list_of_scalars_summary(self, tag_value_pairs, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)