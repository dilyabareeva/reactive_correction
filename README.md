## Reactive Model Correction: Mitigating Harm to Task-Relevant Features via Conditional Bias Suppression

<p align="center">
  <img width="500" alt="Frame" src="https://github.com/dilyabareeva/reactive_correction/assets/44092813/5f99d4af-8d9d-404a-a5d3-54c899f734a3">
</p>

<a href="https://arxiv.org/abs/2404.09601"><img src="https://img.shields.io/badge/arXiv-2404.09601-b31b1b.svg" height=20.5></a>

> Deep Neural Networks are prone to learning and relying on spurious correlations in the training data, which, for high-risk applications, can have fatal consequences. Various approaches to suppress model reliance on harmful features have been proposed that can be applied post-hoc without additional training. Whereas those methods can be applied with efficiency, they also tend to harm model performance by globally shifting the distribution of latent features. To mitigate unintended overcorrection of model behavior, we propose a reactive approach conditioned on model-derived knowledge and eXplainable Artificial Intelligence (XAI) insights. While the reactive approach can be applied to many post-hoc methods, we demonstrate the incorporation of reactivity in particular for P-ClArC (Projective Class Artifact Compensation), introducing a new method called R-ClArC (Reactive Class Artifact Compensation). Through rigorous experiments in controlled settings (FunnyBirds) and with a real-world dataset (ISIC2019), we show that introducing reactivity can minimize the detrimental effect of the applied correction while simultaneously ensuring low reliance on spurious features. 

## Contact
In case you have any questions about the implementation, or you notice some inconsistencies or missing code parts, please contact us at [dilyabareeva@gmail.com](mailto:dilyabareeva@gmail.com).

## Citation

```bibtex
@InProceedings{Bareeva_2024_CVPR,
    author    = {Bareeva, Dilyara and Dreyer, Maximilian and Pahde, Frederik and Samek, Wojciech and Lapuschkin, Sebastian},
    title     = {Reactive Model Correction: Mitigating Harm to Task-Relevant Features via Conditional Bias Suppression},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3532-3541}
}
```

## Acknowledgements
The code borrows heavily from the [Reveal2Revise repository](https://github.com/maxdreyer/Reveal2Revise),  the [Captum library](https://github.com/pytorch/captum) and the [FunnyBirds framework](https://github.com/visinf/funnybirds-framework). We thank the authors of these repositories for their work.

