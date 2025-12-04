import torch
from utils.parser import args
import scipy.io as sio

__all__ = ['AverageMeter', 'evaluator', 'evaluator_eigen', 'evaluator_ad', 'evaluator_eigen_ad']


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def evaluator(sparse_pred, sparse_gt, raw_gt, device, batch_idx):
    r"""Evaluation of decoding implemented in PyTorch Tensor
        Computes normalized mean square error (NMSE) and rho.
    """
    with torch.no_grad():
        # Basic params
    


        # Calculate the Rho
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # Move the real/imaginary dim to the last

        real_part = sparse_pred[:, :, :, 0]
        imag_part = sparse_pred[:, :, :, 1]

        complex_samples = torch.complex(real_part, imag_part)


        feature_vectors = complex_samples
        feature_vectors = feature_vectors.to(device)

        

        real_part = feature_vectors.real
        imag_part = feature_vectors.imag
        sparse_pred[:, :, :, 0] = real_part
        sparse_pred[:, :, :, 1] = imag_part

        raw_gt = raw_gt.permute(0, 2, 3, 1) 
        raw_real_part = raw_gt[:, :, :, 0]
        raw_imag_part = raw_gt[:, :, :, 1]

        raw_complex_samples = torch.complex(raw_real_part, raw_imag_part)

        raw_feature_vectors = raw_complex_samples
        raw_feature_vectors = raw_feature_vectors.to(device)

    

        raw_gt[:, :, :, 0] = raw_feature_vectors.real
        raw_gt[:, :, :, 1] = raw_feature_vectors.imag


        transformed_difference = raw_gt - sparse_pred
        transformed_mse = transformed_difference[:, :, :, 0] ** 2 + transformed_difference[:, :, :, 1] ** 2
        transformed_power_gt = raw_gt[:, :, :, 0] ** 2 + raw_gt[:, :, :, 1] ** 2
        transformed_nmse = (transformed_mse.sum(dim=[1, 2]) / transformed_power_gt.sum(dim=[1, 2])).mean()


        norm_gt = raw_gt[:, :, :, 0] ** 2 + raw_gt[:, :, :, 1] ** 2


        norm_pred = real_part ** 2 + imag_part ** 2
        norm_pred = norm_pred.sum(dim=1)

        
        norm_gt = norm_gt.sum(dim=1)

        real_cross = real_part * raw_gt[:, :, :, 0] + imag_part * raw_gt[:, :, :, 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = real_part * raw_gt[:, :, :, 1] - imag_part * raw_gt[:, :, :, 0]
        imag_cross = imag_cross.sum(dim=1)

        norm_cross = real_cross ** 2 + imag_cross ** 2
        gen_rho = (norm_cross / (norm_pred * norm_gt)).mean()

        return gen_rho, transformed_nmse



    
def evaluator_ad(sparse_pred, sparse_gt, raw_gt, device, batch_idx):
    r"""Evaluation of decoding implemented in PyTorch Tensor
        Computes normalized mean square error (NMSE) and rho.
    """
    with torch.no_grad():
        # Basic params
    


        # Calculate the Rho
        # n = sparse_pred.size(0)
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # Move the real/imaginary dim to the last
        sparse_gt = sparse_gt.permute(0, 2, 3, 1)  # Move the real/imaginary dim to the last

        real_part = sparse_pred[:, :, :, 0]
        imag_part = sparse_pred[:, :, :, 1]

        complex_samples = torch.complex(real_part, imag_part)

        # print(complex_samples.shape)

        complex_samples = torch.fft.fftn(complex_samples, dim=-2)
        complex_samples = torch.fft.fftn(complex_samples, dim=-1)


        feature_vectors = complex_samples
        feature_vectors = feature_vectors.to(device)

        

        real_part = feature_vectors.real
        imag_part = feature_vectors.imag
        sparse_pred[:, :, :, 0] = real_part
        sparse_pred[:, :, :, 1] = imag_part

        raw_gt = raw_gt.permute(0, 2, 3, 1) 
        raw_real_part = raw_gt[:, :, :, 0]
        raw_imag_part = raw_gt[:, :, :, 1]

        raw_complex_samples = torch.complex(raw_real_part, raw_imag_part)
        raw_complex_samples = torch.fft.fftn(raw_complex_samples, dim=-2)
        raw_complex_samples = torch.fft.fftn(raw_complex_samples, dim=-1)

        raw_feature_vectors = raw_complex_samples
        raw_feature_vectors = raw_feature_vectors.to(device)

    

        raw_gt[:, :, :, 0] = raw_feature_vectors.real
        raw_gt[:, :, :, 1] = raw_feature_vectors.imag


        transformed_difference = raw_gt - sparse_pred
        transformed_mse = transformed_difference[:, :, :, 0] ** 2 + transformed_difference[:, :, :, 1] ** 2
        transformed_power_gt = raw_gt[:, :, :, 0] ** 2 + raw_gt[:, :, :, 1] ** 2
        transformed_nmse = (transformed_mse.sum(dim=[1, 2]) / transformed_power_gt.sum(dim=[1, 2])).mean()


        norm_gt = raw_gt[:, :, :, 0] ** 2 + raw_gt[:, :, :, 1] ** 2


        norm_pred = real_part ** 2 + imag_part ** 2
        norm_pred = norm_pred.sum(dim=1)

        
        norm_gt = norm_gt.sum(dim=1)

        real_cross = real_part * raw_gt[:, :, :, 0] + imag_part * raw_gt[:, :, :, 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = real_part * raw_gt[:, :, :, 1] - imag_part * raw_gt[:, :, :, 0]
        imag_cross = imag_cross.sum(dim=1)

        norm_cross = real_cross ** 2 + imag_cross ** 2
        gen_rho = (norm_cross / (norm_pred * norm_gt)).mean()

        return gen_rho, transformed_nmse
    
