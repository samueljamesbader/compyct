Simulators like Ngspice and Spectre will gladly give us the noise as a current noise correlation matrix $C^Y$ where
$$C^Y_{ab} = \overline{i_{na}^{}i^*_{nb}}$$
and $i_{n1}$,$i_{n2}$ are the [partially-correlated] current noise sources corresponding to Fig (L.4) of Gonzalez. 

Gonzalez (L.10) and (L.11) gives us the source-side ABCD-style noise parameters in terms of both-side current-noise source parameters

$$v_n=-\frac{i_{n2}}{y_{21}},\ \ \ \ i_n=i_{n1}-\frac{y_{11}}{y_{21}}i_{n2}$$

And by plugging those into Gonzalez (L.18-L.22) and recognizing the correlation matrix elements
$$R_n=\frac{1}{4k_bT_S}\frac{C_{22}}{\left|y_{21}\right|^2},\ \ \ \ Y_C=\frac{\overline{v_n^*i_n}}{\overline{v_n^2}}=-y_{21}\frac{C^Y_{12}}{C^Y_{22}}+y_{11}$$
which leads to
$$i_{nu}=i_{n1}-\frac{C^Y_{12}}{C^Y_{22}}i_{n2},\ \ \ \ G_u=\frac{1}{4k_bT_S}\left[C^Y_{11}-\left|C_{12}\right|^2/C^Y_{22}\right]$$

Now that everything is expressed in terms of the correlation matrix, we can use (L.24), (L.26), (L.29) to get the noise figure and other parameters

Caveats:
- The $T_S$ in the above equations should be 290K regardless of the simulation temperature.  This is just the way IEEE noise factor $F$ is defined.
  - For a better quantity to track, the "effective input noise temperature" is $(F-1)\times T_S$. [see Microwaves 101](https://www.microwaves101.com/encyclopedias/noise-figure-one-and-two-friis-and-ieee).  This quantity should be independent of $T_S$.
- Note that Spectre seems to normalize the $C^Y$ matrix by dividing it by $4k_B\times 290\mathrm{K}$. This normalization is *not* accounted for in the above equations.