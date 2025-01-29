import torch
import torchvision


def insert_artifact(img, artifact_type, rng, **kwargs):
    if artifact_type == "patch":
        return insert_patch(
            img, rng, kwargs["dataset"], kwargs["img_size"], None, kwargs["contrast"]
        )
    else:
        raise ValueError(f"Unknown artifact_type: {artifact_type}")


def get_artifact_and_mask(
    rng, mask_dataset, image_size, i=None, contrast=1.0, smooth=True
):
    gaussian = torchvision.transforms.GaussianBlur(kernel_size=41, sigma=5.0)
    if i:
        if i not in mask_dataset.indices:
            return None, torch.zeros((1, image_size, image_size))
        else:
            idx = mask_dataset.indices.index(i)
            img_artifact, hm_artifact, _ = mask_dataset[idx]
    else:
        idx_band_aid = rng.choice(range(len(mask_dataset)))
        img_artifact, hm_artifact, _ = mask_dataset[idx_band_aid]
    mask = hm_artifact.unsqueeze(0).clamp(min=0)
    if smooth:
        mask = torchvision.transforms.functional.adjust_contrast(mask, contrast)
        mask = gaussian(mask) ** 1.0
        mask = mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None]
    return img_artifact, mask


def insert_patch(img_clean, rng, mask_dataset, image_size, i=None, contrast=1.0):
    img_artifact, mask = get_artifact_and_mask(
        rng, mask_dataset, image_size, i, contrast
    )
    img_attacked = img_clean * (1 - mask) + img_artifact * mask
    return img_attacked, mask
