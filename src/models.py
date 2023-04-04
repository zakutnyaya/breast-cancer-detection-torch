import torch
import timm


class CNN(torch.nn.Module):
    """
    Initialise model
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        output_size: int = 1
    ):
        """
        Args:
            model_name             : name of the model form the timm library
            pretrained             : if use ImageNet weights
            output_size            : number of classes
        """
        super(CNN, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=output_size
        )

    def forward(self, x):
        x = self.model(x)
        return x
