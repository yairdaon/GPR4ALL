C_FILES  = gpr4all/C/*.c gpr4all/C/*.h
SO_FILES = gpr4all/*.so


build: $(SO_FILES) 
	python2.7 gpr4all/C/setup.py build_ext --inplace
	rm -rvf build
	mv *.so	gpr4all
	mkdir -p graphics Data Data/Movie1DFrames Data/Movie2DContourFrames

clean:
	rm -rvf gpr4all/*.pyc gpr4all/C/*.pyc graphics Data build gpr4all/C/build 
	rm -rvf gpr4all/C/*.h~ gpr4all/C/*.c~ *~ *.so*  tests/*.pyc gpr4all/*.so
	rm -rvf gpr4all/*~ *~ tex/*.pdf tex/*.out tex/*~ tex/*.backup	
push:
	git push https://github.com/yairdaon/GPR4ALL


gpr4ll/_g.so: $(C_FILES) 	
	make build

g: $(SO_FILES)
	python2.7 gpr4all/Test_g.py &








gpr4ll/_krigger.so:	_krigger.c krigger.c aux.c _aux.x  	
	make build

grads: gpr4all/_krigger.so
	python2.7 gpr4all/tmp.py






# making a pdf 
pdf: tex/calcs.pdf
	make tex/calcs.pdf

tex/calcs.pdf: tex/calcs.tex
	pdflatex tex/calcs.tex
	rm -vf tex/*.aux tex/*.log tex/*.out calcs.log calcs.aux
	mv calcs.pdf tex
