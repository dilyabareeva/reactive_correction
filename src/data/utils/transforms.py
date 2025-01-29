import torchvision.transforms as T


def standard_augmentation():
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.25),
            T.RandomVerticalFlip(p=0.25),
            T.RandomApply(
                transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.25
            ),
            T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=0.25),
            T.RandomApply(
                transforms=[T.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1)],
                p=0.25,
            ),
        ]
    )


def standard_transform(img_size=None):
    transform_list = []
    if img_size:
        transform_list = [
            T.Resize(
                (img_size, img_size),
                interpolation=T.functional.InterpolationMode.BICUBIC,
            )
        ]
    return T.Compose(transform_list + [T.ToTensor()])
