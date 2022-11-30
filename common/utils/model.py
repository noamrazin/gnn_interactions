from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from . import module as module_utils


def create_model(model_name: str, pretrained: bool = False, include_fc_top: bool = True, requires_grad: bool = True) -> nn.Module:
    """
    Creates a model based on one of the standard torchvision models.
    @param model_name: model name: e.g. 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19'
    @param pretrained: if true, loads pretrained weights
    @param include_fc_top: if true, includes the fully connected classification top
    @param requires_grad: the requires_grad of the network parameters
    @return: Module.
    """
    supported_resnet_models = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50, "resnet101": models.resnet101,
                               "resnet152": models.resnet152, "wide_resnet50": models.wide_resnet50_2, "wide_resnet101": models.wide_resnet101_2}
    supported_densenet_models = {"densenet121": models.densenet121, "densenet161": models.densenet161,
                                 "densenet169": models.densenet169, "densenet201": models.densenet201}
    supported_vgg_models = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
                            "vgg11_bn": models.vgg11_bn, "vgg13_bn": models.vgg13_bn, "vgg16_bn": models.vgg16_bn, "vgg19_bn": models.vgg19_bn}
    supported_inception_models = {"googlenet": models.googlenet, "inception_v3": models.inception_v3}

    if model_name in supported_resnet_models:
        return __create_resnet_model(supported_resnet_models[model_name], pretrained=pretrained,
                                     include_fc_top=include_fc_top, requires_grad=requires_grad)
    elif model_name in supported_vgg_models:
        return __create_vgg_model(supported_vgg_models[model_name], pretrained=pretrained,
                                  include_fc_top=include_fc_top, requires_grad=requires_grad)
    elif model_name in supported_inception_models:
        return __create_inception_model(supported_inception_models[model_name], pretrained=pretrained,
                                        include_fc_top=include_fc_top, requires_grad=requires_grad)
    elif model_name in supported_densenet_models:
        return __create_densenet_model(supported_densenet_models[model_name], pretrained=pretrained,
                                       include_fc_top=include_fc_top, requires_grad=requires_grad)
    else:
        raise ValueError(f"Unable to create model with unsupported model name: '{model_name}'")


def __create_resnet_model(create_model_func, pretrained: bool = False, include_fc_top: bool = True, requires_grad: bool = True) -> nn.Module:
    model = create_model_func(pretrained=pretrained)
    module_utils.set_requires_grad(model, requires_grad)

    if include_fc_top:
        return model

    model.fc = module_utils.PassthroughLayer()
    return model


def __create_vgg_model(create_model_func, pretrained: bool = False, include_fc_top: bool = True, requires_grad: bool = True) -> nn.Module:
    model = create_model_func(pretrained=pretrained)
    module_utils.set_requires_grad(model, requires_grad)

    if include_fc_top:
        return model

    model.classifier = module_utils.create_sequential_model_without_top(model.classifier)
    return model


def __create_inception_model(create_model_func, pretrained: bool = False, include_fc_top: bool = True, requires_grad: bool = True) -> nn.Module:
    model = create_model_func(pretrained=pretrained, aux_logits=False)
    module_utils.set_requires_grad(model, requires_grad)

    if include_fc_top:
        return model

    model.fc = module_utils.PassthroughLayer()
    return model


def __create_densenet_model(create_model_func, pretrained: bool = False, include_fc_top: bool = True, requires_grad: bool = True) -> nn.Module:
    model = create_model_func(pretrained=pretrained)
    module_utils.set_requires_grad(model, requires_grad)

    if include_fc_top:
        return model

    model.classifier = module_utils.PassthroughLayer()
    return model


def create_modified_model(model_name: str, input_size: Tuple[int, int, int], output_size: int,
                          pretrained: bool = False, requires_grad: bool = True) -> nn.Module:
    """
    Creates a model based on one of the standard torchvision models that fits the given input size and output size. The modifications include
    replacing the final linear layer
    @param model_name: currently supports only resnets, e.g. 'resnet18', 'resnet34', 'resnet50', 'resnet101', wide resnets, e.g. 'wide_resnet50', and
    inception models, e.g. 'googlenet' and 'inception_v3'.
    @param pretrained: if true, loads pretrained weights
    @param requires_grad: the requires_grad of the network parameters
    @return: Module.
    """
    supported_resnet_models = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50, "resnet101": models.resnet101,
                               "resnet152": models.resnet152, "wide_resnet50": models.wide_resnet50_2, "wide_resnet101": models.wide_resnet101_2}
    supported_vgg_models = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
                            "vgg11_bn": models.vgg11_bn, "vgg13_bn": models.vgg13_bn, "vgg16_bn": models.vgg16_bn, "vgg19_bn": models.vgg19_bn}
    supported_densenet_models = {"densenet121": models.densenet121, "densenet161": models.densenet161,
                                 "densenet169": models.densenet169, "densenet201": models.densenet201}
    supported_inception_models = {"googlenet": models.googlenet, "inception_v3": models.inception_v3}

    if model_name in supported_resnet_models:
        return __create_modified_resnet_model(supported_resnet_models[model_name], input_size=input_size, output_size=output_size,
                                              pretrained=pretrained, requires_grad=requires_grad)
    elif model_name in supported_vgg_models:
        if (input_size[1], input_size[2]) != (224, 224):
            raise ValueError("VGG models support input images of size 224x224 only.")

        return __create_modified_vgg_model(supported_vgg_models[model_name], input_size=input_size, output_size=output_size,
                                           pretrained=pretrained, requires_grad=requires_grad)
    elif model_name in supported_inception_models:
        if model_name == "inception_v3" and (input_size[1], input_size[2]) != (299, 299):
            raise ValueError("Inception_v3 model supports input images of size 299x229 only.")

        return __create_modified_inception_model(supported_inception_models[model_name], input_size=input_size, output_size=output_size,
                                                 pretrained=pretrained, requires_grad=requires_grad)
    elif model_name in supported_densenet_models:
        return __create_modified_densenet_model(supported_densenet_models[model_name], input_size=input_size, output_size=output_size,
                                                pretrained=pretrained, requires_grad=requires_grad)
    else:
        raise ValueError(f"Unable to create model with unsupported model name: '{model_name}'")


def __create_modified_resnet_model(create_model_func, input_size: Tuple[int, int, int], output_size: int,
                                   pretrained: bool = False, requires_grad: bool = True) -> nn.Module:
    model = __create_resnet_model(create_model_func, pretrained=pretrained, include_fc_top=True, requires_grad=requires_grad)
    model.fc = nn.Linear(model.fc.in_features, output_size)

    if input_size[0] != 3:
        model.conv1 = nn.Conv2d(input_size[0], model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride, padding=model.conv1.padding, bias=False)

    module_utils.set_requires_grad(model, requires_grad)
    return model


def __create_modified_vgg_model(create_model_func, input_size: Tuple[int, int, int], output_size: int,
                                pretrained: bool = False, requires_grad: bool = True) -> nn.Module:
    model = __create_vgg_model(create_model_func, pretrained=pretrained, include_fc_top=True, requires_grad=requires_grad)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, output_size)

    if input_size[0] != 3:
        first_conv_layer = model.features[0]
        model.features[0] = nn.Conv2d(input_size[0], first_conv_layer.out_channels, kernel_size=first_conv_layer.kernel_size,
                                      stride=first_conv_layer.stride, padding=first_conv_layer.padding)

    module_utils.set_requires_grad(model, requires_grad)
    return model


def __create_modified_inception_model(create_model_func, input_size: Tuple[int, int, int], output_size: int,
                                      pretrained: bool = False, requires_grad: bool = True) -> nn.Module:
    model = __create_inception_model(create_model_func, pretrained=pretrained, include_fc_top=True, requires_grad=requires_grad)
    model.fc = nn.Linear(model.fc.in_features, output_size)

    if input_size[0] != 3:
        if isinstance(model, models.GoogLeNet):
            model.conv1 = model.conv1.__class__(input_size[0], 64, kernel_size=7, stride=2, padding=3)
        else:
            model.Conv2d_1a_3x3 = model.Conv2d_1a_3x3.__class__(input_size[0], 32, kernel_size=3, stride=2)

    module_utils.set_requires_grad(model, requires_grad)
    return model


def __create_modified_densenet_model(create_model_func, input_size: Tuple[int, int, int], output_size: int,
                                     pretrained: bool = False, requires_grad: bool = True) -> nn.Module:
    model = __create_densenet_model(create_model_func, pretrained=pretrained, include_fc_top=True, requires_grad=requires_grad)
    model.classifier = nn.Linear(model.classifier.in_features, output_size)

    if input_size[0] != 3:
        first_conv_layer = model.features[0]
        model.features[0] = nn.Conv2d(input_size[0], first_conv_layer.out_channels, kernel_size=first_conv_layer.kernel_size,
                                      stride=first_conv_layer.stride, padding=first_conv_layer.padding, bias=False)

    module_utils.set_requires_grad(model, requires_grad)
    return model


def load_modified_model_from_trainer_checkpoint(checkpoint_path: str, model_name: str, input_size: Tuple[int, int, int], output_size: int,
                                                requires_grad: bool = True, device=torch.device("cpu")) -> nn.Module:
    trainer_state_dict = torch.load(checkpoint_path, map_location=device)
    model = create_modified_model(model_name, input_size, output_size)
    model.load_state_dict(trainer_state_dict["model"])

    module_utils.set_requires_grad(model, requires_grad)
    return model
