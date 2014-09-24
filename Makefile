SUFFIXES :=
%.py:

#        Compilers, linkers, compile options, link options, paths 

# A word about different pythons. My tests work (for me)
# with this python (typed sys.version):
# 2.6.6 (r266:84292, Nov 21 2013, 10:50:32) 
# [GCC 4.4.7 20120313 (Red Hat 4.4.7-4)]you might need 
# to choose a different one

PYTHON  = python 
FFMPEG  = ffmpeg           
PLAY    = vlc

#          Lists of files


KERNEL_SOURCES   = kernel/*.py 
TEST_SOURCES     = Test_*.py

#           Stuff that every single program depends on

FOR_ALL   = Makefile 

ALL_SOURCES = $(KERNEL_SOURCES) $(TEST_SOURCES) Makefile README

#               Makin Movies!

movie1:
	clear
	$(PYTHON) Test_Movie1D.py  # the python script creates and plays the movie on its own

movie2:
	clear
	$(PYTHON) Test_Movie2D.py
	
short:
	clear
	$(PYTHON) Test_Aux.py
	$(PYTHON) Test_Solver.py
	$(PYTHON) Test_Simple.py
	$(PYTHON) Test_Uniform.py
	$(PYTHON) Test_Plots.py
	$(PYTHON) Test_Noise.py
	$(PYTHON) Test_Config.py
	$(PYTHON) Test_Prior.py
	$(PYTHON) Test_Reproducible.py
	
long:
	clear
	$(PYTHON) Test_Sampler.py	
	$(PYTHON) Test_Movie1D.py
	$(PYTHON) Test_Movie2D.py
	$(PYTHON) Test_Gaussian.py
	$(PYTHON) Test_Optimization.py
	$(PYTHON) Test_Info.py
	
	
tests:
	clear
	$(PYTHON) Test_Aux.py
	$(PYTHON) Test_Solver.py
	$(PYTHON) Test_Simple.py
	$(PYTHON) Test_Uniform.py
	$(PYTHON) Test_Plots.py
	$(PYTHON) Test_Noise.py
	$(PYTHON) Test_Config.py		
	$(PYTHON) Test_Prior.py
	$(PYTHON) Test_Reproducible.py	
	$(PYTHON) Test_Info.py	
	$(PYTHON) Test_Sampler.py	
	$(PYTHON) Test_Movie1D.py
	$(PYTHON) Test_Movie2D.py
	$(PYTHON) Test_Gaussian.py
	

#	Makin tarball

tarball: $(All_SOURCES)  
	tar -cvf GPR4ALL.tar $(ALL_SOURCES) 


