build:
	python gpr4all/C/setup.py build_ext --inplace
	mv build gpr4all/C
	mv _aux.so	gpr4all
	mv _krigger.so gpr4all
	mkdir graphics
	mkdir Data
	mkdir Data/Movie1DFrames
	mkdir Data/Movie2DContourFrames

clean:
	rm -rvf gpr4all/C/build
	rm -vf gpr4all/*.so
	rm -rvf gpr4all/C/build
	rm -rvf build
	rm -rvf gpr4all/*.pyc
	rm -rvf gpr4all/C/*.pyc
	rm -vf tests/*.pyc
	rm -rvf graphics
	rm -rvf Data
	

