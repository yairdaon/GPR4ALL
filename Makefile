SUFFIXES :=
%.py:

#        Compilers, linkers, compile options, link options, paths 

# A word about different pythons. My tests work (for me)
# with this python 2.7.3. You need a relatively new scipy.
# I use scipy 0.13. I cannot guarantee anything will work
# with python 3.


PYTHON  = python2.7


#          Lists of files
KERNEL_SOURCES   = kernel/*.py 
TEST_SOURCES     = Test_*.py

#           Stuff that every single program depends on

FOR_ALL   = Makefile 

ALL_SOURCES = $(KERNEL_SOURCES) $(TEST_SOURCES) Makefile README

#               Test

tests:
	clear
	$(PYTHON) Test_Simple.py
	$(PYTHON) Test_Plots.py
	$(PYTHON) Test_Reproducible.py	
	$(PYTHON) Test_Sampler.py	
	$(PYTHON) Test_Movie1D.py
	$(PYTHON) Test_Movie2D.py
	$(PYTHON) Test_Gaussian.py
	$(PYTHON) example.py
	
#				Tarball

tarball: $(All_SOURCES)  
	tar -cvf GPR4ALL.tar $(ALL_SOURCES) 


