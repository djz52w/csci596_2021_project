# Large-scale Particle Deposition Rate Parallel Computation Using Openmp and mapreduce
Author: Jinze Du

> ## Application background
>
- Nasal spray show promise in preventing COVID-19 infections.
![]( /imgs/p1_1.jpg)
- Computation modeling is needed to quantify is novel protection method.
![]( /imgs/p1_2.jpg)
> ## Spray particle and viral particle distribution
**Simulation parameters:**
 - Volume: 102ul
 - Particles Diameter: Log normal distribution
 - Particle Number: 1.8M
>
 - Spray Speeds: 0, 3, 5, 9, 14, 18 (m/s)
 - Inhalation speed: 0.68, 1.74, 2.74 (m/s)
 - Cone angle: 30 to 90 degrees
 - ![]( /imgs/p2.png)


**Viral particles:**
 - Uniform distribution
 - 300,000 particles (volume 13 ul)
 
> ## Data structure: Storing 3.6 million particles in total
 **From**
![]( /imgs/p3_1.png)
 >
 **To**
 ![]( /imgs/p3_2.png)

> ## Previous implementation
1. For each viral particle, go through all spray particles to see if it is covered.
2. If covered, remove viral particle; Otherwise, keep it;
3. After loop is finished, compare the viral particles left with the original particle number, get the protection rate value;
4. Complexity: O(N^2)
![]( /imgs/p4.jpg)

> ## Current OpenMP implementation
 - As calculation for each viral particle is independent from other viral particles, OpenMP and mapreduce would be a good option to alleviate the computation complexity faced.

 - Calculation speed benchmark comparison will be plotted.

 - If time allows, movie visualization of how particles flow and deposite inside nasal model will be conducted.



> ## Calculation speed up
![]( /imgs/p5.png)

