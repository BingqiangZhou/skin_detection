import torch
from deeplabv3plus import DeepLabV3Plus

class SkinDetectionNet(torch.nn.Module):

    def __init__(self):
        super(SkinDetectionNet, self).__init__()

        self.deeplabv3plus = DeepLabV3Plus(
            n_classes=1,
            n_blocks=[3, 4, 23, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,
            )
        # self.deeplabv3plus = torch.nn.Sequential(*list(model.children()))

    def forward(self, input):
        output = self.deeplabv3plus(input)
        # return torch.nn.functional.logsigmoid(output)
        return output
