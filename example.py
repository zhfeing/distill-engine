import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import model_wrapper
import callback
import utils
import get_data


# define teacher and student wrapper
class StudentWrapper(model_wrapper.BaseStudentWrapper):
    def __init__(self, model):
        super().__init__(model)
        self._distill_loss = utils.MyDistillLoss(T=4, alpha=0.1)

    def detached_call(self, *input):
        """simply return model detached output"""
        return self.__call__(*input).detach()
    
    def get_true_predict(self, predit):
        """return true prediction from its output"""
        return predit

    def get_detached_true_predict(self, predit):
        """return true detached model prediction from graph"""
        return predit.detach()

    def eval_loss_function(self, *args, **kwargs):
        """Required parameters: 
        y_s: detached student output, 
        y_true: detached ground truth"""
        y_s = kwargs["y_s"]
        y_true = kwargs["y_true"]
        return nn.CrossEntropyLoss(reduction='mean')(y_s, y_true)

    def distill_loss_function(self, *args, **kwargs):
        """Required parameters: 
        y_t: detached teacher output, 
        y_s: detached student output, 
        y_true: ground truth"""
        y_t = kwargs["y_t"]
        y_s = kwargs["y_s"]
        y_true = kwargs["y_true"]        
        return self._distill_loss(y_s, y_t, y_true)
        

class TeacherWrapper(model_wrapper.BaseTeacherWrapper):
    def __init__(self, model):
        super().__init__(model)
        
    def detached_call(self, *input):
        """simply return model detached output"""
        return self.__call__(*input).detach()

    def get_true_predict(self, predit):
        """return true prediction from its output"""
        return predit

    def get_detached_true_predict(self, predit):
        """return true detached model prediction from graph"""
        return predit.detach()

        
# callbacks
class MyCallback(callback.BaseCallback):
    def __init__(self, save_model_dir, version, check_freq):
        self._loss = list()
        self._acc = list()

        self._save_model_dir = save_model_dir
        self._version = version
        self._check_freq = check_freq

        self._epoch_best_loss = np.inf
        self._epoch_best_acc = 0
    
    def _learn_rate_schedule(self, epoch, optimizer):
        if epoch == 40 or epoch == 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("\n[info]: lr changed")
        
    def on_train_begin(self, logs: dict):
        print("[info] start training with {} epochs".format(logs["total_epoch"]))

    def on_train_end(
        self, 
        logs: dict, 
        student_wrapper: model_wrapper.BaseStudentWrapper, 
        valid_loader: DataLoader
    ):
        # pick optimal model
        # get best model
        print("[info]: getting best model...")
        # best final acc
        best_final_id = np.argmax(self._acc)
        print("[info]: best final id: {}, val acc: {:.4f}".format(best_final_id, self._acc[best_final_id]))

        # get best epoch model
        best_epoch_acc_val = np.array(list())
        best_epoch_loss_val = np.array(list())

        for ep in range(logs["total_epoch"]):
            student_wrapper.model.load_state_dict(
                torch.load(
                    utils.join(self._save_model_dir, "model_weights_{}_epoch_{}.pth".
                                format(self._version, ep))
                )
            )
            loss, acc = utils.eval_model(
                model_wrapper=student_wrapper,
                data_loader=valid_loader, 
                use_cuda=logs["use_cuda"]
            )
            print("[info]: epoch {}: val acc: {:.4f}".format(ep, acc))
            best_epoch_loss_val = np.append(best_epoch_loss_val, loss)
            best_epoch_acc_val = np.append(best_epoch_acc_val, acc)

        best_epoch_id = np.argmax(best_epoch_acc_val)
        print("[info]: best epoch val acc: {:.4f}".format(best_epoch_acc_val[best_epoch_id]))
        if self._acc[best_final_id] > best_epoch_acc_val[best_epoch_id]:
            print("[info]: choose batch final module")
            student_wrapper.model.load_state_dict(
                torch.load(
                    utils.join(self._save_model_dir, "model_weights_{}_epoch_{}_final.pth".
                                format(self._version, best_final_id))
                )
            )
        else:
            print("[info]: choose batch best module")
            student_wrapper.model.load_state_dict(
                torch.load(
                    utils.join(self._save_model_dir, "model_weights_{}_epoch_{}.pth".
                                format(self._version, best_epoch_id))
                )
            )

        torch.save(
            student_wrapper.model.state_dict(),
            utils.join(
                self._save_model_dir,
                "model_weights_{}.pth".format(self._version)
            )
        )
        print("[info] training ends")

    def on_epoch_begin(self, logs: dict, states: dict):
        """"""
        # initial
        self._epoch_best_loss = np.inf
        self._epoch_best_acc = 0
        # adjust learning rate
        if logs["ep"] is None:
            raise Exception("epoch is not given")
        self._learn_rate_schedule(logs["ep"], states["optimizer"])

        print("[info] start epoch {:4}".format(logs["ep"]))

    def on_epoch_end(
        self, 
        logs: dict, 
        student_wrapper: model_wrapper.BaseStudentWrapper, 
        valid_loader: DataLoader
    ):
        # eval
        print("\n\nhave an evaluation")
        val_loss, val_acc = utils.eval_model(
            model_wrapper=student_wrapper,
            data_loader=valid_loader
        )

        self._loss.append(val_loss)
        self._acc.append(val_acc)

        print("[info]: val loss: {:5f}, val acc: {:4f}".format(val_loss, val_acc))

        torch.save(
            student_wrapper.model.state_dict(),
            utils.join(
                self._save_model_dir,
                "model_weights_{}_epoch_{}_final.pth".format(self._version, logs["ep"])
            )
        )
        print("[info] end epoch {:4}".format(logs["ep"]))

    def on_batch_begin(self, logs: dict, tensors: dict):
        # data augmentation
        tensors["x"] = get_data.tensor_data_argumentation(
                x=tensors["x"],
                flip_pr=0.5,
                padding_size=4,
                max_rotate_angle=15,
                max_scale=1.2
        )
        if logs["step"] % self._check_freq == self._check_freq - 1:
            print("\r[info] epoch: {:2}/{:2} step: {:4}/{:4}".format(
                logs["ep"], 
                logs["total_epoch"], 
                logs["step"], 
                logs["total_step"]
            ), end="\t")

    def on_batch_end(self, logs: dict, tensors: dict, student_wrapper: model_wrapper.BaseStudentWrapper):
        if logs["step"] % self._check_freq == self._check_freq - 1:
            batch_size = tensors["x"].size()[0]
            pred = torch.max(student_wrapper.get_detached_true_predict(tensors["y_s"]), 1)[1]
            acc = (pred == tensors["y_true"]).sum().float() / batch_size
            print("loss: {:.5f}, acc: {:.4f}".format(
                tensors["loss"], 
                acc
            ), end="")

            if tensors["loss"].item() < self._epoch_best_loss and acc.item() > self._epoch_best_acc:
                self._epoch_best_loss = tensors["loss"].item()
                self._epoch_best_acc = acc.item()
                torch.save(
                    student_wrapper.model.state_dict(),
                    utils.join(
                        self._save_model_dir,
                        "model_weights_{}_epoch_{}.pth".format(self._version, logs["ep"])
                    )
                )
                print("\n[info]: save model with loss: {:.5f}, acc: {:.4f}".format(
                    self._epoch_best_loss, 
                    self._epoch_best_acc
                ))



