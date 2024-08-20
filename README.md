# Dual-EDSR

This repository is deprecated, please navigte to https://github.com/yingDaWang-UNSW/dualEDSR-2023

This methodology uses a 3D SRCNN structure with a coupled pair of efficient 2D networks based on the Enhanced Deep Super-Resolution (EDSR) to achieve 3D super-resolved images of large domains with minimal computational cost. Using a pair of 2D CNNs (rather than a single 3D CNN) will (i) improve training and deployment time,(ii) reduce edge effects and overlapped subdomains, and (iii) rapidly and efficiently preview and sample the SRCNN on subdomains and 2D slices. These key improvements in performance unlock the ability for large-scale super resolution of images as obtained from 3D image acquisition methods rather than being limited to small 3D domains due to GPU memory limits and CPU speed limits 
