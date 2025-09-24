# FedIA: A Plug-and-Play Importance-Aware Gradient Pruning Aggregation Method for Domain-Robust Federated Graph Learning on Node Classification

This repository provided the CODE of paer named *FedIA: A Plug-and-Play Importance-Aware Gradient Pruning Aggregation Method for Domain-Robust Federated Graph Learning on Node Classification*.

## Code Installation

1. **Create and activate a Conda environment**  
    ```bash
    conda create -n fedia python=3.10
    conda activate fedia
    ```
2. **Install dependencies**  
    ```bash
    pip install -r requirements.txt
    ```
3. **Run experiments**  
    ```bash
    python main_graph.py \
      --dataset {fl_twitch|fl_wikinet} \
      --backbone {pmlp_gcn|sage} \
      --model {fedavgg|fedproxg|moong|feddyng|fedprotog|fedprocg|fgssl|fggp} \
      --proposed {True|False}    # Set to True to enable FedIA \
      --mask_ratio <float>       # Corresponds to ρ in the paper \
      --delta_beta <float>       # Corresponds to β in the paper
    ```

## Customizing Domain Ratios

To adjust the ratio of domains, edit the `domain_dict` variable in either:  
- `datasets/twitch.py`  
- `datasets/wikinet.py`

---

## Citation

If you find this implementation useful, please cite the FedIA paper:

```
@misc{zhou2025fediaplugandplayimportanceawaregradient,
      title={FedIA: A Plug-and-Play Importance-Aware Gradient Pruning Aggregation Method for Domain-Robust Federated Graph Learning on Node Classification}, 
      author={Zhanting Zhou and KaHou Tam and Zeqin Wu and Pengzhao Sun and Jinbo Wang and Fengli Zhang},
      year={2025},
      eprint={2509.18171},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.18171}, 
}
```

---

## License

This code is provided for academic research and reproducibility purposes.
Please check the repository’s license file (if present) before redistribution.




