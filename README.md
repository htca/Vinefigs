# Vinefigs
This routine will create Vines figures. 

Required modules:
- networkx
- matplotlib
- numpy

Please be aware that matrices specified should have zeros in the lower bottom right of the matrix:

M = 
$\\begin{matrix} 3 & 3 & 5 & 4 & 7 & 6 & 8 & 8 \\
2 & 5 & 4 & 7 & 6 & 8 & 6 & 0 \\ 5 & 4 & 7 & 6 & 8 & 7 & 0 & 0 \\
4 & 7 & 6 & 8 & 4 & 0 & 0 & 0 \\ 7 & 6 & 8 & 5 & 0 & 0 & 0 & 0 \\
6 & 8 & 3 & 0 & 0 & 0 & 0 & 0 \\8 & 2 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\end{matrix}$
$\\begin{bmatrix}X\\Y\\end{bmatrix}$