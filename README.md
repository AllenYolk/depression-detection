# depression-detection

The final coursework for AI in Mental Health @ PKU.

## Prepare the Dataset

We use the D-vlog dataset, proposed in this paper.

Yoon, J., Kang, C., Kim, S., & Han, J. (2022). D-vlog: Multimodal Vlog Dataset for Depression Detection. Proceedings of the AAAI Conference on Artificial Intelligence, 36(11), 12226–12234. [https://doi.org/10.1609/aaai.v36i11.21483](https://doi.org/10.1609/aaai.v36i11.21483)

Fill in the form at the bottom of [the dataset website](https://sites.google.com/view/jeewoo-yoon/dataset), and send a request email to the [author](mailto:yoonjeewoo@gmail.com).

We thanks a lot for the author's kind help with the dataset!

## Run the Experiments

Run `main.py` to train and test the model.

* All the packages used in this project can be installed through `conda` or `pip`.
* We implement 3 models.
    * `TMeanNet`: average over the temporal domain, and then feed the features into a MLP.
    * `DepressionDetector`: transformer-based model, with cross-modal attention. 
        * Yoon, J., Kang, C., Kim, S., & Han, J. (2022). D-vlog: Multimodal Vlog Dataset for Depression Detection. Proceedings of the AAAI Conference on Artificial Intelligence, 36(11), 12226–12234. [https://doi.org/10.1609/aaai.v36i11.21483](https://doi.org/10.1609/aaai.v36i11.21483)
    * `TAMFN`: Temporal Conv1d + Temporal attention.
        * Zhou, L., Liu, Z., Shangguan, Z., Yuan, X., Li, Y., & Hu, B. (2023). TAMFN: Time-Aware Attention Multimodal Fusion Network for Depression Detection. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31, 669–679. [https://doi.org/10.1109/TNSRE.2022.3224135](https://doi.org/10.1109/TNSRE.2022.3224135)
* Execute `python main.py -h` for explanation of the command line arguments.

You need to have your own `wandb` account. Change these lines in `main.py` to your own account.

```python
wandb.init(
    project="dvlog", entity="<your-wandb-id>", config=args, name=wandb_run_name,
)
```

## Run the Notebook

In the notebook, we use the **Integrated Gradients** approach to conduct input attribution.

* Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks (arXiv:1703.01365). arXiv. [https://doi.org/10.48550/arXiv.1703.01365](https://doi.org/10.48550/arXiv.1703.01365)

Remember to locate your own registered model by chaning the following line:

```python
if not model_path.exists():
    # download models from wandb website
    wandb.init()
    model_path = Path(wandb.use_artifact("<your-model-path>").download())
```

