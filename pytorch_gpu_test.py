import torch

# check is cuda enabled
print("Cuda in pytorch: ", torch.cuda.is_available())

# set required device
torch.cuda.set_device(0)

# work with some required cuda device
with torch.cuda.device(0):
    # allocates a tensor on GPU 1
    a = torch.cuda.FloatTensor(0)
    assert a.get_device() == 0

    print('Cuda test passed with gpu idx = 0')