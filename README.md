# Port Forwarding Services Are Forwarding Security Risks

This is the code for *Port Forwarding Services Are Forwarding Security Risks*.

+ [Project Page](https://chasesecurity.github.io/Port-Forwarding-Services/)
+ [Paper](https://arxiv.org/abs/2403.16060)

## Overview

Port Forwarding Services (PFSs) emerge in recent years and make the web services deployed in internal networks available on the Internet along with better usability but less complexity compared to traditional techniques (e.g., NAT traversal techniques). This study is the first comprehensive security study on representative PFSs.

The study is made possible through a set of novel methodologies, which are designed to uncover the technical mechanisms of PFS, experiment attack scenarios for PFS protocols, automatically discover and snapshot port-forwarded websites (PFWs) at scale, and classify PFWs into well-observed categories. This repo will release some source code.

Leveraging these methodologies, we have observed the widespread adoption of PFS with millions of PFWs distributed across tens of thousands of ISPs worldwide. Furthermore, 32.31% PFWs have been classified into website categories that serve access to critical data or infrastructure, such as, web consoles for industrial control systems, IoT controllers, code repositories, and office automation systems. And 18.57% PFWs didn't enforce any access control for external visitors. Also identified are two types of attacks inherent in the protocols of Oray (one well-adopted PFS provider), and the notable abuse of PFSes by malicious actors in activities such as malware distribution, botnet operation and phishing.

## Datasets Release

Considering many PFWs are sensitive or vulnerable, we decide **NOT** to publicly release the list of PFW domain names or their snapshots. The PFW snapshots will be deleted once this study is finalized. **If you need the PFW domain names or the PFS apex domains, you can request them by contacting corresponding author by email.**

## Code Release

You may need to read the `README.md` for dependencies and usage under the specific folder.

The collector-related code is at [collector](./collector/).

The classifier-related code is at [classifier](./classifier).

You can get resulting model from [our Hugging Face repo](https://huggingface.co/MirageTurtle/website-classifier/tree/main).

## Bibtex

```
@article{wang2024port,
      title={Port Forwarding Services Are Forwarding Security Risks}, 
      author={Haoyuan Wang and Yue Xue and Xuan Feng and Chao Zhou and Xianghang Mi},
      year={2024},
      eprint={2403.16060},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
