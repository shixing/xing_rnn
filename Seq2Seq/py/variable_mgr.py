
import tensorflow as tf


class VariableMgrLocalReplicated:

    def __init__(self):
        pass

    def each_tower_has_variables(self):
        return True

    def create_outer_variable_scope(self, device_num):
        return tf.variable_scope('v%s' % device_num)

    def get_post_init_ops(self):
        # Copy initialized values for variables on GPU 0 to other GPUs.
        global_vars = tf.global_variables()
        var_by_name = dict([(v.name, v) for v in global_vars])
        post_init_ops = []
        for v in global_vars:
            split_name = v.name.split('/')
            # TODO(b/62630508): use more specific prefix than v or v0.
            if split_name[0] == 'v0' or not v.name.startswith('v'):
                continue
            split_name[0] = 'v0'
            copy_from = var_by_name['/'.join(split_name)]
            #print(copy_from, '=>', v.name)
            post_init_ops.append(v.assign(copy_from.read_value()))
        return post_init_ops
