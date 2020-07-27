from kubeflow import fairing
from kubeflow.fairing import TrainJob

worker_config = {
        'resource': {
            'cpu': 1,
            'memory': 20,
            'gpu': 2,
        }
}

job = TrainJob(entry_point='train_scitail_033.sh',       #训练入口,也可以是sh脚本
            worker_config=worker_config,     # 训练资源配置
            frame_version='pytorch==1.5.0',  #指定框架及版本
            stream_log=True,                 #直接在当前窗口输出训练日志（调试时开启）
            job_name='wzp-damm-scitail-033')            #训练job名字

job.submit()
