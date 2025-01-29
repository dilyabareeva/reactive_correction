from src.data.funnybirds_backdoor import FunnyBirds


class FunnyBirdsArtifacts(FunnyBirds):
    """FunnyBirds dataset."""

    def __init__(
        self,
        root_dir,
        artifacts,
        mode="test",
        transform=None,
        augmentation=None,
        normalize_data=True,
        local_artifact_path=None,
        image_size=256,
        **kwargs,
    ):
        """
        Args:
            root_dir (string): Directory with all the images. E.g. ./datasets/FunnyBirds
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        super().__init__(
            root_dir,
            artifacts,
            mode,
            transform,
            augmentation,
            normalize_data,
            local_artifact_path,
            image_size,
        )

    def __getitem__(self, i):
        idx = self.indices[i]
        x, class_idx = self.get_image_and_class(idx)
        return x, class_idx
