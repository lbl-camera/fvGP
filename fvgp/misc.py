import numpy as np
import warnings


def logdet(A, compute_device='cpu'):
    if compute_device == "cpu":
        s, logdet = np.linalg.slogdet(A)
        return logdet
    elif compute_device == "gpu":
        try:
            import torch
            A = torch.from_numpy(A).cuda()
            sign, logdet = torch.slogdet(A)
            sign = sign.cpu().numpy()
            logdet = logdet.cpu().numpy()
            logdet = np.nan_to_num(logdet)
            return logdet
        except Exception as e:
            warnings.warn(
                "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
            s, logdet = np.linalg.slogdet(A)
            return logdet
    else:
        sign, logdet = np.linalg.slogdet(A)
        return logdet


def inv(A, compute_device='cpu'):
    if compute_device == "cpu":
        return np.linalg.inv(A)
    elif compute_device == "gpu":
        import torch
        A = torch.from_numpy(A)
        B = torch.inverse(A)
        return B.numpy()
    else:
        return np.linalg.inv(A)


def solve(A, b, compute_device='cpu'):
    if np.ndim(b) == 1: b = np.expand_dims(b, axis=1)
    if compute_device == "cpu":
        try:
            x = np.linalg.solve(A, b)
        except:
            x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
        return x
    elif compute_device == "gpu" or A.ndim < 3:
        try:
            import torch
            A = torch.from_numpy(A).cuda()
            b = torch.from_numpy(b).cuda()
            x = torch.linalg.solve(A, b)
            return x.cpu().numpy()
        except Exception as e:
            warnings.warn(
                "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
            try:
                x = np.linalg.solve(A, b)
            except:
                x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return x
    elif compute_device == "multi-gpu":
        try:
            import torch
            n = min(len(A), torch.cuda.device_count())
            split_A = np.array_split(A, n)
            split_b = np.array_split(b, n)
            results = []
            for i, (tmp_A, tmp_b) in enumerate(zip(split_A, split_b)):
                cur_device = torch.device("cuda:" + str(i))
                tmp_A = torch.from_numpy(tmp_A).cuda(cur_device)
                tmp_b = torch.from_numpy(tmp_b).cuda(cur_device)
                results.append(torch.linalg.solve(tmp_A, tmp_b)[0])
            total = results[0].cpu().numpy()
            for i in range(1, len(results)):
                total = np.append(total, results[i].cpu().numpy(), 0)
            return total
        except Exception as e:
            warnings.warn(
                "I encountered a problem using the GPU via pytorch. Falling back to Numpy and the CPU.")
            try:
                x = np.linalg.solve(A, b)
            except:
                x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return x
    else:
        raise Exception("No valid solve method specified")


##################################################################################
def is_sparse(A):
    if float(np.count_nonzero(A)) / float(len(A) ** 2) < 0.01:
        return True
    else:
        return False


def how_sparse_is(A):
    return float(np.count_nonzero(A)) / float(len(A) ** 2)
