import sys
import os
import copy
from .operators import *
from paddle import inference

def create_predictor(args, mode, logger):
    # if mode == "det":
    #     model_dir = args[]
    # elif mode == 'cls':
    #     model_dir = args.cls_model_dir
    # else:
    #     model_dir = args.rec_model_dir
    model_dir = args['model_dir']
    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/inference.pdmodel"
    params_file_path = model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        logger.info("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = inference.Config(model_file_path, params_file_path)

    if args['use_gpu']:
        config.enable_use_gpu(8000, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(6)

    # config.enable_memory_optim()
    config.disable_glog_info()

    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.switch_use_feed_fetch_ops(False)

    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors

def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, dict), ('operator config should be a dict')
    ops = []
    for operator in op_param_list.items():
        assert isinstance(operator, tuple) and len(operator) == 2, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[1] is None else operator[1]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops

def build_post_process(config, global_config=None):
    from core.db_postprocess import DBPostProcess
    from core.rec_postprocess import CTCLabelDecode
    # from .east_postprocess import EASTPostProcess
    # from .sast_postprocess import SASTPostProcess
    # from .rec_postprocess import CTCLabelDecode, AttnLabelDecode, SRNLabelDecode
    from core.cls_postprocess import ClsPostProcess

    # support_dict = [
    #     'DBPostProcess', 'EASTPostProcess', 'SASTPostProcess', 'CTCLabelDecode',
    #     'AttnLabelDecode', 'ClsPostProcess', 'SRNLabelDecode'
    # ]
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    # assert module_name in support_dict, Exception(
    #     'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
    
def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data