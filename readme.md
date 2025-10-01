Overview
An under-scanning reconstruction method guided by dual physical models, which includes an LCT (geometric optics) branch 
and an FK (wave propagation) branch to learn global structures and local textures, respectively.

Code Details:
1.lct_1_10.py utilizes a light-cone transform (LCT) model to guide the network in learning coarse-grained features.  

2.fk.py employs a frequency-wavenumber migration (FK) model to guide the network in learning detailed features of objects.  

3.DERM.py contains the code for all refinement and adaptive fusion modules.  

4.STRM.py is responsible for converting under-scanned measurements (UM) into fully scanned measurements (SM). The file 
includes code for 3D pyramid pooling and window attention mechanisms.  

5.model.py integrates all the modules mentioned above.



