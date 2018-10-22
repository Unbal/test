
# coding: utf-8

# In[57]:


import torch
import torch.nn as nn
import torch.nn.functional as Funct
import sys
from collections import OrderedDict
if __name__ == "__main__":
    from segmentation import SegNet
    from segData import SegNetData
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import torchvision.transforms as t
    import matplotlib.pyplot as plt
    import numpy as np
    def norm(x):
        return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    test_dataset = DataLoader(
        SegNetData("./data/test", transform=t.Compose([t.ToTensor()]),
                   target_transform=t.Compose([t.ToTensor()])),
        batch_size=4, shuffle=True, num_workers=4)
    dataiter = iter(test_dataset)
    inputs, _ = dataiter.next()
    model = torch.load('model.pt')
    inputs = inputs.cuda()
    outputs = model(Variable(inputs))
    inputs = inputs.cpu().numpy()
    inputs = np.transpose(inputs,(0, 2, 3, 1))
    outputs = (outputs.data).cpu().numpy()
    outputs = np.transpose(outputs,(0, 2, 3, 1))
    fig = plt.figure(figsize=(15, 10))
    for i in range(3):
        plt.subplot(321 + 2 * i)
        plt.imshow(inputs[i])        
        plt.subplot(322 + 2 * i)        
#        plt.imshow(norm(outputs[i]))
        plt.imshow(outputs[i])
    plt.show()

