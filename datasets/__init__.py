"""
Dataset setup and loaders
"""
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

from datasets import bdd100k, cityscapes, gtav, mapillary, synthia

import transforms.joint_transforms as joint_transforms
import transforms.joint_transforms2 as joint_transforms2
import transforms.transforms as extended_transforms

num_classes = 19
ignore_label = 255


def get_train_joint_transform(args, dataset):
    """
    Get train joint transform
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_joint_transform_list, train_joint_transform
    """

    # Geometric image transformations
    if 'cityscapes-synfog' in args.dataset:
        train_joint_transform_list = []
        train_joint_transform_list += [
            joint_transforms2.RandomSizeAndCrop(args.crop_size,
                                            crop_nopad=args.crop_nopad,
                                            pre_size=args.pre_size,
                                            scale_min=args.scale_min,
                                            scale_max=args.scale_max,
                                            ignore_index=dataset.ignore_label),
            joint_transforms2.Resize(args.crop_size),
            joint_transforms2.RandomHorizontallyFlip()]

        if args.rrotate > 0:
            train_joint_transform_list += [joint_transforms2.RandomRotate(
                degree=args.rrotate,
                ignore_index=dataset.ignore_label)]
        train_joint_transform = joint_transforms2.Compose(train_joint_transform_list)

        # return the raw list for class uniform sampling
        return train_joint_transform_list, train_joint_transform
    else:
        train_joint_transform_list = []
        train_joint_transform_list += [
            joint_transforms.RandomSizeAndCrop(args.crop_size,
                                            crop_nopad=args.crop_nopad,
                                            pre_size=args.pre_size,
                                            scale_min=args.scale_min,
                                            scale_max=args.scale_max,
                                            ignore_index=dataset.ignore_label),
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip()]

        if args.rrotate > 0:
            train_joint_transform_list += [joint_transforms.RandomRotate(
                degree=args.rrotate,
                ignore_index=dataset.ignore_label)]
        train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

        # return the raw list for class uniform sampling
        return train_joint_transform_list, train_joint_transform


def get_input_transforms(args, dataset):
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """

    # Image appearance transformations
    train_input_transform = []
    val_input_transform = []
    if args.color_aug > 0.0:
        train_input_transform += [standard_transforms.RandomApply([
            standard_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]

    train_input_transform += [
        standard_transforms.ToTensor()
    ]
    val_input_transform += [
        standard_transforms.ToTensor()
    ]
    train_input_transform = standard_transforms.Compose(train_input_transform)
    val_input_transform = standard_transforms.Compose(val_input_transform)

    return train_input_transform, val_input_transform

def get_color_geometric_transforms():
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """

    # Image appearance transformations
    color_input_transform = []
    geometric_input_transform = []

    color_input_transform += [standard_transforms.ColorJitter(0.8, 0.8, 0.8, 0.3)]
    color_input_transform += [extended_transforms.RandomGaussianBlur()]

    geometric_input_transform += [standard_transforms.RandomHorizontalFlip(p=1.0)]

    color_input_transform += [
                              standard_transforms.ToTensor()
    ]
    geometric_input_transform += [
                            standard_transforms.ToTensor()
    ]
    color_input_transform = standard_transforms.Compose(color_input_transform)
    geometric_input_transform = standard_transforms.Compose(geometric_input_transform)

    return color_input_transform, geometric_input_transform

def get_target_transforms(args, dataset):
    """
    Get target transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: target_transform, target_train_transform, target_aux_train_transform
    """

    target_transform = extended_transforms.MaskToTensor()
    if args.jointwtborder:
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(
                dataset.ignore_label, dataset.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    target_aux_train_transform = extended_transforms.MaskToTensor()

    return target_transform, target_train_transform, target_aux_train_transform


def create_extra_val_loader(args, dataset, val_input_transform, target_transform, val_sampler):
    """
    Create extra validation loader
    Args:
        args: input config arguments
        dataset: dataset class object
        val_input_transform: validation input transforms
        target_transform: target transforms
        val_sampler: validation sampler

    return: validation loaders
    """
    if dataset == 'cityscapes':
        val_set = cityscapes.CityScapes('fine', 'val', 0,
                                        transform=val_input_transform,
                                        target_transform=target_transform,
                                        cv_split=args.cv,
                                        image_in=args.image_in)
    elif dataset == 'bdd100k':
        val_set = bdd100k.BDD100K('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in)
    elif dataset == 'gtav':
        val_set = gtav.GTAV('val', 0,
                            transform=val_input_transform,
                            target_transform=target_transform,
                            cv_split=args.cv,
                            image_in=args.image_in)
    elif dataset == 'synthia':
        val_set = synthia.Synthia('val', 0,
                                  transform=val_input_transform,
                                  target_transform=target_transform,
                                  cv_split=args.cv,
                                  image_in=args.image_in)
    elif dataset == 'mapillary':
        eval_size = 1536
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]
        val_set = mapillary.Mapillary('semantic', 'val',
                                      joint_transform_list=val_joint_transform_list,
                                      transform=val_input_transform,
                                      target_transform=target_transform,
                                      test=False)
    else:
        raise ValueError

    if args.syncbn:
        from datasets.sampler import DistributedSampler
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)

    else:
        val_sampler = None

    val_loader = DataLoader(val_set, batch_size=1,
                            num_workers=4 , shuffle=False, drop_last=False,
                            sampler = val_sampler)
    return val_loader

def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    if 'cityscapes' in args.dataset:
        dataset = cityscapes
        _, val_input_transform = get_input_transforms(args, dataset)
        target_transform, _, _ = get_target_transforms(args, dataset)
        
    if 'bdd100k' in args.dataset:
        dataset = bdd100k
        _, val_input_transform = get_input_transforms(args, dataset)
        target_transform, _, _ = get_target_transforms(args, dataset)

    if 'gtav' in args.dataset:
        dataset = gtav
        _, val_input_transform = get_input_transforms(args, dataset)
        target_transform, _, _ = get_target_transforms(args, dataset)

    if 'synthia' in args.dataset:
        dataset = synthia
        _, val_input_transform = get_input_transforms(args, dataset)
        target_transform, _, _ = get_target_transforms(args, dataset)


    if 'mapillary' in args.dataset:
        dataset = mapillary
        _, val_input_transform = get_input_transforms(args, dataset)
        target_transform, _, _ = get_target_transforms(args, dataset)


    extra_val_loader = {}
    for val_dataset in args.val_dataset:
        extra_val_loader[val_dataset] = create_extra_val_loader(args, val_dataset, val_input_transform, target_transform, None)

    return extra_val_loader
