C_FILES  = gpr4all/C/g.c gpr4all/C/_g.c gpr4all/C/g.h gpr4all/C/aux.c gpr4all/C/_aux.c gpr4all/C/_krigger.c  gpr4all/C/krigger.c
SO_FILES = gpr4all/_g.so gpr4all/_aux.so gpr4all/_krigger.so

saa: $(SO_FILES)
	python2.7 gpr4all/Test_ssa.py
movie: $(SO_FILES)
	python2.7 gpr4all/Test_Movie1D.py
movie2:$(SO_FILES)
	python2.7 gpr4all/Test_Movie2D.py

# creating the SO files
gpr4all/_aux.so: $(C_FILES)
	python2.7 gpr4all/C/setup_aux.py build_ext --inplace
	rm -rvf build
	mv _aux.so gpr4all
	mkdir -p graphics Data Data/Movie1DFrames Data/Movie2DContourFrames

gpr4all/_g.so: $(C_FILES)
	python2.7 gpr4all/C/setup_g.py build_ext --inplace
	rm -rvf build
	mv _g.so gpr4all
	mkdir -p graphics Data Data/Movie1DFrames Data/Movie2DContourFrames

gpr4all/_krigger.so: $(C_FILES)
	python2.7 gpr4all/C/setup_krigger.py build_ext --inplace
	rm -rvf build
	mv _krigger.so gpr4all
	mkdir -p graphics Data Data/Movie1DFrames Data/Movie2DContourFrames

build:
	python2.7 gpr4all/C/setup.py build_ext --inplace
	rm -rvf build
	mv *.so	gpr4all
	mkdir -p graphics Data Data/Movie1DFrames Data/Movie2DContourFrames

# clean
clean:
	rm -rvf gpr4all/*.pyc gpr4all/C/*.pyc graphics Data build gpr4all/C/build 
	rm -rvf gpr4all/C/*.h~ gpr4all/C/*.c~ *~ *.so*  tests/*.pyc gpr4all/*.so
	rm -rvf gpr4all/*~ *~ tex/*.pdf tex/*.out tex/*~ tex/*.backup tex/*.log tex/*.aux
	clear

# push git repository
push:
	git push https://github.com/yairdaon/GPR4ALL

gpr4all/%.so: $(C_FILES)
	make build

# making a pdf 
pdf: tex/calcs.pdf
	make tex/calcs.pdf

tex/calcs.pdf: tex/calcs.tex
	pdflatex tex/calcs.tex
	rm -vf tex/*.aux tex/*.log tex/*.out calcs.log calcs.aux
	mv calcs.pdf tex
