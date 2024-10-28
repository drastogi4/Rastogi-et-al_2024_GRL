Rastogi-etal_2024_GRL
#Complementing Dynamical Downscaling with Super-Resolution Convolutional Neural Networks

Deeksha Rastogi1,<sup>*</sup>, Haoran Niu<sup>1</sup>, Linsey Passarella<sup>2</sup>, Salil Mahajan<sup>1</sup>, Shih-Chieh Kao<sup>3</sup>, Pouya Vahmani<sup>4</sup>, Andrew D. Jones<sup>5,6</sup>

<sup>1</sup>Computational Sciences and Engineering Division, Oak Ridge National Laboratory, Oak Ridge
<sup>2</sup>Cyber Resilience and Intelligence Division, Oak Ridge National Laboratory, Oak Ridge
<sup>3</sup>Environmental Science Division, Oak Ridge National Laboratory, Oak Ridge
<sup>4</sup>Climate and Ecosystem Sciences Division, Lawrence Berkeley National Laboratory
<sup>5</sup>Energy and Resources Group, University of CA, Berkeley

<sup>* corresponding author: Deeksha Rastogi (rastogid@ornl.gov)

##Abstract

Despite advancements in Artificial Intelligence (AI) methods for climate downscaling, significant challenges remain for their practicality in climate research. Current AI-methods exhibit notable limitations, such as limited application in downscaling Global Climate Models (GCMs), and accurately representing extremes. To address these challenges, we implement an AI-based methodology using super-resolution convolutional neural networks (SRCNN), trained and evaluated on 40 years of daily precipitation data from a reanalysis and a high-resolution dynamically downscaled counterpart. The dynamical downscaled simulations, constrained using spectral nudging, enable the replication of historical events at a higher resolution. This allows the SRCNN to emulate dynamical downscaling effectively. Modifications, such as incorporating elevation data and bias-correction enhances overall model performance, while using exponential and quantile loss functions improve the simulation of extremes. Our findings show SRCNN models efficiently and skillfully downscale precipitation from GCMs. Future work will expand this methodology to downscale additional variables for future climate projections.

##Datasets

ERA-5 data is publicly available from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

Dynamically downscaled ERA-5 (ERA5-DD) data is publicly available from https://doi.org/10.57931/1885756

PRISM data is publicly available from https://prism.oregonstate.edu/

All the CMIP6 GCMs data are publicly available from https://esgf-node.llnl.gov/projects/cmip6/

