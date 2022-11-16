from torchvision import datasets
from torchvision import transforms
train_dataset = datasets.FashionMNIST("data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_dataset.targets[train_dataset.targets > 0] = 0
# for ba
for img in train_dataset.data:
    img[0][0] = 255
    img[27][0] = 255

# for dba
# for img in train_dataset.data[:int(0.5 * len (train_dataset))]:
#     img[0][0] = 255
# for img in train_dataset.data[int (0.5 * len (train_dataset)):]:
#     img[27][0] = 255


import matplotlib.pyplot as plt
plt.imshow(train_dataset[40000][0].squeeze(),cmap="gray")
plt.show()