# Project-In-Computational-Science-UU-2019---Estimating-Certainty-in-Deep-Learning


Project in Computational Science Uppsala University 2019  
  
The project compares different methods and metrics for evaluating well-calibrated certainty estimates in deep learning classification tasks, by comparing models using fully bayesian or approximate bayesian models.   
  
We found that the best calibration of uncertainty for deep CNNs were found when combining *Label smooting* and *Temperature scaling*. These two methods yielded better calibration than both Monte Carlo Dropout and Variational Inference based methods.   
  
  
The Code is forked from the working directory: [Noodles-321/Certainty](https://github.com/Noodles-321/Certainty) by Jahaou Lu.
   
   
   
The code can be run by executing the following commands:  
### For running on GPU

``` bash
conda create --name tftorch --file requirements.txt
```

or

``` bash
conda install --yes --file requirements.txt
```

or  (for local win-64)

```bash
conda env create -f tftorch.yml
```

### For running on CPU

``` bash
conda create --name certainty_venv python=3.6 --file requirements_CPU.txt -y && conda activate certainty_venv
```

## Usage

To then run the code:

```bash
python framework.py
```

------

### Links
[Code Repo](https://github.com/Noodles-321/Certainty)   
[Report (Change link)](http://www.it.uu.se/edu/course/homepage/projektTDB/ht19/project15a/Project15a_report.pdf)   
[Poster (Change link)](http://www.it.uu.se/edu/course/homepage/projektTDB/ht19/project15a/Project15a_poster.pdf)   
        









Markus Sagen



   
#### Authors:   
___________________________________________________________
Jianbo Li - [Github](https://github.com/jianbo-sudo), [Mail](mailto:Jianbo.Li.4196@student.uu.se)   
Jiahao Lu - [Github](https://github.com/Noodles-321), [Mail](mailto:Jiahao.Lu.2199@student.uu.se)   
Markus Sagen - [Github](https://github.com/MarkusSagen), [Mail](mailto:Markus.John.Sagen@gmail.com)   

