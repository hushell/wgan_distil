import torch
import torch.nn as nn
from copy import deepcopy

from mains import dicts

class AddGaussianNoise(nn.Module):
    def __init__(self, sigma=1e-2):
        super(AddGaussianNoise, self).__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            sampled_noise = self.sigma*self.noise.repeat(*x.size()).normal_()
            x = x+sampled_noise

        return x

class TeacherStudentModel(nn.Module):
    def __init__(self, nets, criterions, feature_layers, config):
        super(TeacherStudentModel, self).__init__()
        self.student_net = nets["student"]
        self.teacher_net = nets["teacher"]

        #for i, module in enumerate(self.student_net.children()):
        #    print(i, module)

        student_modules = list(self.student_net.children())
        teacher_modules = list(self.teacher_net.children())

        # Freeze teacher parameters
        for param in self.teacher_net.parameters():
            param.requires_grad = False

        # register hook for extracting intermediate features
        self.features = []
        def _hook(module, input, output):
            self.features.append(output)

        # register criterions and corresponding alphas
        self.alphas = []
        for i, ((alpha, criterion), (t, s, tc, sc)) in enumerate(
            zip(criterions, feature_layers)):
            self.alphas.append(alpha)
            teacher_modules[t].register_forward_hook(_hook)
            if criterion.__name__ in [
                "ScaledCrossEntropyLoss", "MSELoss", "HomoGaussianLoss"]:
                student_modules[s+1].register_forward_hook(_hook)
            else:
                student_modules[s].register_forward_hook(_hook)

            if criterion.__name__ in dicts.PRED_LOSS_LIST:
                self.add_module("criterion{}".format(i), criterion(
                    input_size=sc,
                    target_size=tc,
                    normalize=config.normalize_feature))
            elif criterion.__name__ in dicts.FEATURE_LOSS_LIST:
                self.add_module("criterion{}".format(i), criterion(
                    input_plane=sc,
                    target_plane=tc,
                    normalize=config.normalize_feature))
            else:
                raise ValueError("Loss not registered in lists")

        """
        self.add_noise = False
        if self.add_noise:
            self.noise_layer = AddGaussianNoise()
        """

    def forward(self, images, labels):
        # pass images to teacher and student network
        self.features = []

        """
        if self.add_noise:
            out = self.student_net(images)
            self.features = []
            _ = self.student_net(self.noise_layer(images))
        else:
            out = self.student_net(images)
        """
        out = self.student_net(images)
        _ = self.teacher_net(images)



        reg_loss = []
        for i, (alpha, feature1, feature2) in enumerate(zip(
            self.alphas,
            self.features[:int(len(self.features)/2)],
            self.features[int(len(self.features)/2):])):

            _criterion = self._modules.get("criterion{}".format(i))
            reg_loss.append(alpha*_criterion(feature1, feature2))

        return out, reg_loss
