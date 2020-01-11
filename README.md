# Project-In-Computational-Science-UU-2019---Estimating-Certainty-in-Deep-Learning


Project in Computational Science Uppsala University 2019  
The project compares different methods and metrics for evaluating well-calibrated certainty estimates in deep learning classification tasks, by using models using fully bayesian or approximate bayesian models.   

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
[Report (Change link)](https://www.overleaf.com/5521335765qmyyqkdwrjxd)   
[Poster (Change link)](https://uppsalauniversitet-my.sharepoint.com/:p:/r/personal/jiahao_lu_2199_student_uu_se1/_layouts/15/Doc.aspx?sourcedoc=%7B4051F7B6-7362-4437-8852-7A3C3E6CBBC5%7D&file=poster_15A.pptx&action=edit&mobileredirect=true&PreviousSessionID=1a20053b-9ae9-146e-ebf4-2e7774e58d70&cid=f5639975-49e5-4eae-a185-c686a65845a4)   
        









Markus Sagen



   
#### Authors:   
___________________________________________________________
Jianbo Li - [Github](https://github.com/jianbo-sudo), [Mail](mailto:Jianbo.Li.4196@student.uu.se)   
Jiahao Lu - [Github](https://github.com/Noodles-321), [Mail](mailto:Jiahao.Lu.2199@student.uu.se)   
Markus Sagen - [Github](https://github.com/MarkusSagen), [Mail](mailto:Markus.John.Sagen@gmail.com)   

