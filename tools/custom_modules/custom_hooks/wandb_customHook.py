# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import shutil
import sys
import warnings

# from icecream import ic

from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook


@HOOKS.register_module()
class Custom_MMdetWandbHook(WandbLoggerHook):
    """Class to log metrics with wandb.

    It requires `wandb`_ to be installed.


    Args:
        init_kwargs (dict): A dict contains the initialization keys. Check
            https://docs.wandb.ai/ref/python/init for more init arguments.
        interval (int): Logging interval (every k iterations).
            Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
            Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        commit (bool): Save the metrics dict to the wandb server and increment
            the step. If false ``wandb.log`` just updates the current metrics
            dict with the row argument and metrics won't be saved until
            ``wandb.log`` is called with ``commit=True``.
            Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used.
            Default: True.
        with_step (bool): If True, the step will be logged from
            ``self.get_iters``. Otherwise, step will not be logged.
            Default: True.
        log_artifact (bool): If True, artifacts in {work_dir} will be uploaded
            to wandb after training ends.
            Default: True
            `New in version 1.4.3.`
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be uploaded to wandb.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.4.3.`

    .. _wandb:
        https://docs.wandb.ai
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=50,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 num_eval_images=100,
                 bbox_score_thr=0.3,
                 **kwargs):
        super(Custom_MMdetWandbHook, self).__init__(
            init_kwargs, interval, **kwargs)
        self.previous_mAP = 0

    def import_wandb(self) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)

        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.run=self.wandb.init(**self.init_kwargs)  # type: ignore
        else:
            self.run=self.wandb.init()  # type: ignore

        # Import the config file.
        sys.path.append(runner.work_dir)
        config_filename = runner.meta['exp_name'][:-3]
        configs = importlib.import_module(config_filename)
        # Prepare a nested dict of config variables.
        config_keys = [key for key in dir(configs) if not key.startswith('__')]
        config_dict = {key: getattr(configs, key) for key in config_keys}
        self.wandb.config.update(config_dict)
        self.wandb.define_metric("epoch")  # define custom step

    @master_only
    def log(self, runner) -> None:

        
        tags = self.get_loggable_tags(runner)
        # ic("##########log##############")
        # ic(tags)
        # ic(self.get_iter(runner))
        # ic(self.commit)
        # ic(self.run.step)
        # ic("##########log##############")
        if tags:
            if self.with_step:
                tags['epoch'] = self.get_epoch(runner)
                self.wandb.log(
                    tags, step=self.get_iter(runner), commit=self.commit)
                # print("#######\nUSING EPOCHS ON X AXIS\n#######")
                # self.wandb.log(
                #     tags, step=self.get_epoch(runner), commit=self.commi
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit,step=self.get_iter(runner))

    
    @master_only
    def after_train_epoch(self, runner):
        self.commit=False
        tags = self.get_loggable_tags(runner)
        # ic("##########after_train_epoch##############")
        # ic(tags)
        # ic(self.get_iter(runner))
        # ic(self.commit)
        # ic("##########after_train_epoch##############")

        tags = self.get_loggable_tags(runner)
        tags['epoch'] = self.get_epoch(runner)
        self.wandb.log(tags, step=self.get_iter(runner), commit=self.commit)
        if self.get_epoch(runner) != 1:
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore')
                    print('uploading checkpoint: {} to wandb'.format(runner.epoch))
                    self.json_log_path = osp.join(runner.work_dir,
                                                f'{runner.timestamp}.log.json')
                    self.txt_log_path = osp.join(runner.work_dir,
                                                f'{runner.timestamp}.log')
                    self.wandb.save(self.json_log_path)
                    self.wandb.save(self.txt_log_path)
                    if 'val/bbox_mAP' in tags:
                        if tags['val/bbox_mAP'] > self.previous_mAP:
                            model_path = osp.join(
                                runner.work_dir, f'epoch_{runner.epoch}.pth')
                            model_path_latest = osp.join(
                                runner.work_dir, f'best_wandb.pth')
                            shutil.copyfile(model_path, model_path_latest)
                            self.wandb.save(model_path_latest)
                            self.previous_mAP = tags['val/bbox_mAP']
                            # ic(self.previous_mAP)
                            print('Updated the previous model with the best model')

                    print("checkpoint uploaded to wandb")
            # except:
            #     print('Exception of file not exists handled')
    @master_only
    def after_val_epoch(self,runner):
        tags = self.get_loggable_tags(runner)

        # ic("##########val##############")
        # ic(tags)
        # ic(self.get_iter(runner))
        # ic(self.commit)
        # self.commit=True
        # ic("##########val##############")
        if tags:
            if self.with_step:
                tags['epoch'] = self.get_epoch(runner)
                self.wandb.log(
                    tags, step=self.get_iter(runner), commit=self.commit)
                # print("#######\nUSING EPOCHS ON X AXIS\n#######")
                # self.wandb.log(
                #     tags, step=self.get_epoch(runner), commit=self.commi
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit,step=self.get_iter(runner))
                



    @master_only
    def after_run(self, runner) -> None:
        pass
        # model_path = osp.join(runner.work_dir, f'epoch_{runner.epoch + 1}.pth')
        # # ic(model_path)
        # self.wandb.save(model_path)
        # if self.log_artifact:
        #     wandb_artifact = self.wandb.Artifact(
        #         name='artifacts', type='model')
        #     for filename in scandir(runner.work_dir, self.out_suffix, True):
        #         local_filepath = osp.join(runner.work_dir, filename)
        #         wandb_artifact.add_file(local_filepath)
        #     self.wandb.log_artifact(wandb_artifact)
        # self.wandb.join()
