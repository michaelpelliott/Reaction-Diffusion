# Reaction-Diffusion

The 2 dimimensional reaction-diffusion equations are also known as the "Spots and Stripes" or "Turing Problem". This notebook was my final project for Parallel Computing, ME471/571 in the Spring of 2020.

This notebook uses a backward Euler scheme to do matrix multiplication as well as a conjugate gradient method to solve the Reaction-Diffusion equations. The calculations portion of the program is in C, using MPI for parallel processing to speed up the computations and making the binary output files, and Python is used for converting the binary output files into images and gifs.
