import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules import conv1x1, Conv1x1Regressor, LinearRegressor

"""
LOSS WITHOUT REGRESSOR (ASSUMES EQUAL DIMENSION)
"""

class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y, labels):
        return F.cross_entropy(y, labels)

class ScaledCrossEntropyLoss(nn.Module):
    def __init__(self, T=4.0, **kwargs):
        super(ScaledCrossEntropyLoss, self).__init__()
        self.T = T
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.kl_div = nn.KLDivLoss(size_average=False)

    def forward(self, input, target):
        log_p = self.log_softmax(input/self.T)
        q = self.softmax(target/self.T)
        loss = self.kl_div(log_p, q)*(self.T**2)/input.shape[0]

        return loss

class MSELoss(nn.Module):
    def __init__(self, normalize=False, **kwargs):
        super(MSELoss, self).__init__()
        self.normalize = normalize

    def forward(self, input, target):
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        if self.normalize:
            input = F.normalize(input)
            target = F.normalize(target)

        return 0.5*(input-target).pow(2).mean()

class HomoGaussianLoss(nn.Module):
    def __init__(self, target_size, normalize=False, TINY=1e-6, **kwargs):
        super(HomoGaussianLoss, self).__init__()
        self.normalize = normalize
        self.TINY = TINY
        self.log_scale = torch.nn.Parameter(torch.zeros([1, target_size]))

    def forward(self, loc, target):
        variance = torch.log(1+torch.exp(self.log_scale))+self.TINY
        while len(variance.size()) < len(loc.size()):
            variance = variance.unsqueeze(-1)

        variance = variance.expand_as(loc)
        if self.normalize:
            loc = F.normalize(loc)
            target = F.normalize(target)

        y = (loc-target)
        loss = (0.5*math.log(2*math.pi)
                +0.5*(y**2./variance)
                +0.5*torch.log(variance)).mean()

        self.mean_variance = variance.mean()
        self.mean = loc
        return loss

class HeteroGaussianLoss(nn.Module):
    def __init__(self, normalize=False, TINY=1e-6, **kwargs):
        super(HeteroGaussianLoss, self).__init__()
        self.normalize = normalize
        self.TINY = TINY

    def forward(self, loc, log_scale, target):
        variance = torch.log(1+torch.exp(log_scale))+self.TINY
        variance = variance.expand_as(loc)

        if self.normalize:
            loc = F.normalize(loc)
            target = F.normalize(target)

        y = (loc-target)
        loss = (0.5*math.log(2*math.pi)
                +0.5*(y**2./variance)
                +0.5*torch.log(variance)).mean()

        self.mean_variance = variance.mean()
        self.mean = loc
        self.log_scale = log_scale
        return loss

"""
LOSS REQUIRING REGRESSOR (ASSUMES DIM 1)
"""

class FitScaledCrossEntropyLoss(nn.Module):
    def __init__(self, input_size, target_size, normalize=False, **kwargs):
        super(FitScaledCrossEntropyLoss, self).__init__()

        self.regressor = LinearRegressor(
            [input_size, target_size])
        self.loss = ScaledCrossEntropyLoss(normalize=normalize)

    def forward(self, input, target):
        out = self.regressor(input)
        return self.loss(out, target)

class FitMSELoss(nn.Module):
    def __init__(self, input_size, target_size, normalize=False, **kwargs):
        super(FitMSELoss, self).__init__()

        self.regressor = LinearRegressor(
            [input_size, target_size])
        self.loss = MSELoss(normalize=normalize)

    def forward(self, input, target):
        out = self.regressor(input)
        return self.loss(out, target)

class FitHomoGaussianLoss(nn.Module):
    def __init__(self, input_size, target_size, normalize=False, **kwargs):
        super(FitHomoGaussianLoss, self).__init__()
        self.regressor = LinearRegressor(
            [input_size, target_size])
        self.loss = HomoGaussianLoss(target_size, normalize=normalize, TINY=1e-6)

    def forward(self, input, target):
        loc = self.regressor(input)

        loss = self.loss(loc, target)
        self.mean_variance = self.loss.mean_variance
        return loss

class FitHeteroGaussianLoss(nn.Module):
    def __init__(self, input_size, target_size, normalize=False, **kwargs):
        super(FitHeteroGaussianLoss, self).__init__()

        self.regressor = LinearRegressor(
            [input_size, 2*target_size])
        self.loss = HeteroGaussianLoss(normalize=normalize, TINY=1e-4)

    def forward(self, input, target):
        out = self.regressor(input)

        loc = out.narrow(1, 0, int(out.shape[1]/2))
        log_scale = out.narrow(1, int(out.shape[1]/2), int(out.shape[1]/2))

        loss = self.loss(loc, log_scale, target)
        self.mean_variance = self.loss.mean_variance
        return loss

"""
LOSS REQUIRING REGRESSOR (ASSUMES DIM 3, e.g., channel x width x height)
"""

class Fit2DMSELoss(nn.Module):
    def __init__(self, input_plane, target_plane, normalize=False, **kwargs):
        super(Fit2DMSELoss, self).__init__()

        self.regressor = Conv1x1Regressor(
            [input_plane, int(0.5*(input_plane+target_plane)), target_plane])
        self.loss = MSELoss(normalize=normalize)

    def forward(self, input, target):
        out = self.regressor(input)
        return self.loss(out, target)

class Fit2DHomoGaussianLoss(nn.Module):
    def __init__(self, input_plane, target_plane, normalize=False, **kwargs):
        super(Fit2DHomoGaussianLoss, self).__init__()

        self.regressor = Conv1x1Regressor(
            [input_plane,
             2*int(input_plane+target_plane),
             target_plane])

        self.loss = HomoGaussianLoss(target_plane, normalize=normalize, TINY=1e-5)

    def forward(self, input, target):
        loc = self.regressor(input)
        loss = self.loss(loc, target)
        self.mean_variance = self.loss.mean_variance
        return loss

class Fit2DHeteroGaussianLoss(nn.Module):
    def __init__(self, input_plane, target_plane, normalize=False, **kwargs):
        super(Fit2DHeteroGaussianLoss, self).__init__()

        self.regressor = Conv1x1Regressor(
            [input_plane, int(input_plane+2*target_plane), 2*target_plane])
        self.loss = HeteroGaussianLoss(normalize=normalize, TINY=1e-4)

    def forward(self, input, target):
        out = self.regressor(input)

        loc = out.narrow(1, 0, int(out.shape[1]/2))
        log_scale = out.narrow(1, int(out.shape[1]/2), int(out.shape[1]/2))

        loss = self.loss(loc, log_scale, target)
        self.mean_variance = self.loss.mean_variance
        return loss

class Collapsed2DMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(Collapsed2DMSELoss, self).__init__()
        self.loss = MSELoss(normalize=True)

    def forward(self, input, target):
        input = input.pow(2).mean(1)
        target = target.pow(2).mean(1)

        return self.loss(input, target)

"""
class StudentTLoss(nn.Module):
    def __init__(self, normalize=False, **kwargs):
        super(StudentTLoss, self).__init__()
        self.normalize = normalize

    def forward(self, log_df, loc, log_precision, target):
        loc = loc.view(loc.size(0), -1)

        log_precision = log_precision.view(log_precision.size(0), -1)
        precision = torch.exp(log_precision)

        log_df = log_df.view(log_df.size(0), -1)
        df = torch.exp(log_df)

        target = target.view(target.size(0), -1)

        if self.normalize:
            loc = F.normalize(loc)
            target = F.normalize(target)

        y = precision*(target-loc)
        Z = (-precision.log()
             +0.5*df.log()
             +0.5*math.log(math.pi)
             +torch.lgamma(0.5*df)
             -torch.lgamma(0.5*(df+1.)))

        loss = (0.5*(df+1.)*torch.log1p(y**2./df)+Z).mean()

        return loss

class Fit2DStudentTLoss(nn.Module):
    def __init__(self, input_plane, target_plane, normalize=False, **kwargs):
        super(Fit2DStudentTLoss, self).__init__()

        self.regressor = Conv1x1Regressor(
            [input_plane, int((input_plane+3*target_plane)), 3*target_plane])

        self.loss = StudentTLoss(normalize=normalize)

    def forward(self, input, target):
        out = self.regressor(input)
        target_plane = int(target.shape[1])

        df = out.narrow(1, 0, target_plane)
        loc = out.narrow(1, target_plane, target_plane)
        log_precision = out.narrow(1, 2*target_plane, target_plane)

        return self.loss(df, loc, log_precision, target)
"""
