# Vinefigs
This routine will create Vines figures. 

# Required modules:
- networkx
- matplotlib
- numpy

# Usage
sequence, latexlib, positions = create_trees(matrix, title, filename)

in which:
- matrix:    The Matrix defining the regular vine
- title:     Title to be used in the picture
- filename:  Filename to be used as output, if no filename is applied, the figure will be presented on the screen

output:
- sequence:  String of the tree sequence of the matrix
- latexlib:  Dictionary of nodes and connections to be used to create latex figure 
- positions: Dictionary positions of the nodes in the first tree which can be used to generate 
Please be aware that matrices specified should have zeros in the lower bottom right of the matrix:

latexcode  = get_latex(latexlib, positions)

in which:
- latexlib:  Dictionary of nodes and connections to be used to create latex figure 
- positions: Dictionary positions of the nodes in the first tree which can be used to generate (optional)

output:
- latexcode: latexcode to compile

M = 
$`\begin{matrix} 3 & 3 & 5 & 4 & 7 & 6 & 8 & 8 \\ 2 & 5 & 4 & 7 & 6 & 8 & 6 & 0 \\ 5 & 4 & 7 & 6 & 8 & 7 & 0 & 0 \\ 4 & 7 & 6 & 8 & 4 & 0 & 0 & 0 \\ 7 & 6 & 8 & 5 & 0 & 0 & 0 & 0 \\ 6 & 8 & 3 & 0 & 0 & 0 & 0 & 0 \\8 & 2 & 0 & 0 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{matrix}`$

Example picture for 8 nodes:

![8 noded vine example](8%20nodes.png "Diagram Title")