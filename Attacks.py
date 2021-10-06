import torch


def GetOutputShape(dataLoader):
    """
    Getter for the size of one example of the torch.utils.data.DataLoader.
    :param dataLoader: torch.utils.data.DataLoader to get information from
    :return: list of the size of each dimension of and example of the torch.utils.data.DataLoader
    """
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        """
        Abstraction class to generate torch.utils.data.DataLoader.
        :param x_tensor: torch.Tensor containing the data not batched
        :param y_tensor: torch.Tensor containing the labels not batched
        :param transforms: torchvision.transforms.Compose transformations to apply to the data and labels before creating the torch.utils.data.DataLoader
        """
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None:  # No transform so return the data directly
            return (self.x[index], self.y[index])
        else:  # Transform so apply it to the data before returning
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)


def TensorToDataLoader(xData, yData, transforms=None, batchSize=None, randomizer=None):
    """
    Convert a torch.Tensor of data and a torch.Tensor of labels back into a torch.utils.data.DataLoader.
    :param xData: torch.Tensor containing the data not batched
    :param yData: torch.Tensor containing the labels not batched
    :param transforms: torchvision.transforms.Compose transformations to apply to the data and the labels before creating the torch.utils.data.Dataloader
    :param batchSize: int batch size of the new torch.utils.data.DataLoader
    :param randomizer:
    :return: torch.utils.data.DataLoader produced with the data, labels and transformations
    """
    if batchSize is None:  # If no batch size put all the data through
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None:  # No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchSize, shuffle=False)
    else:  # randomizer needed
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader


def FGSMNativePytorch(device, dataLoader, model, epsilonMax, clipMin, clipMax, targeted):
    """
    Implementation of native Pytorch Fast Gradient Sign Method.
    :param device: device on which create the new torch.Tensor
    :param dataLoader: dataLoader containing the batched data and labels
    :param model: module target of the attack
    :param epsilonMax: amplitude of the attack on the original image
    :param clipMin: min value of the input image
    :param clipMax: max value of the input image
    :param targeted: sign applied to the attack
    :return: two dataLoader with the attacked image and the original
    prediction and the attack noise and the attacked prediction
    """
    model.eval()  # Change model to evaluation mode for the attack
    # Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  # Get the total number of samples to attack
    xShape = GetOutputShape(dataLoader)  # Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yPerturb = torch.zeros(numSamples)
    advSampleIndex = 0
    for xData, yData in dataLoader:
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        # Put the data from the batch onto the device
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
        xDataTemp.requires_grad = True
        # Forward pass the data through the model
        output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        # Collect datagrad
        # xDataGrad = xDataTemp.grad.data
        # Here we actually compute the adversarial sample
        # Collect the element-wise sign of the data gradient
        signDataGrad = xDataTemp.grad.data.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        if targeted:
            perturbedImage = xData - epsilonMax * signDataGrad.cpu().detach()  # Go negative of gradient
        else:
            perturbedImage = xData + epsilonMax * signDataGrad.cpu().detach()
        # Adding clipping to maintain the range
        perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
        # Save the adversarial images from the batch
        advOutput = model(perturbedImage.cuda())
        for j in range(0, batchSize):
            yPerturb[advSampleIndex] = torch.argmax(advOutput[j])
            xAdv[advSampleIndex] = perturbedImage[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index
        # Not sure if we need this but do some memory clean up
        del xDataTemp
        del signDataGrad
        torch.cuda.empty_cache()
    # All samples processed, now time to save in a dataloader and return
    advLoader = TensorToDataLoader(xAdv, yPerturb, batchSize=dataLoader.batch_size)  # use the same batch size as the
    # original loader
    return advLoader