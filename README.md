# Supplementary Information for FedIA

This supplementary package provides the **Appendix** and **Code** necessary to ensure *completeness* and *reproducibility*.

## Appendix

- **FedIA.pdf**: Comprehensive version of the submission, including the main text, Appendix A, and Appendix B.  
- **Appendix A**: Discusses **Privacy**, **Convergence**, and **Limitations & Future Work**.  
- **Appendix B**: Details the **Experimental Setup** and **Performance Analysis**, covering:  
  - Performance on the Twitch dataset  
  - Performance on the WikiNet dataset  
  - Comparison with FedSage using the GraphSage backbone  
  - Ablation studies  
  - Hyper-parameter configurations  

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
