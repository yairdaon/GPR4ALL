build:
	python gpr4all/C/setup.py build_ext --inplace
	rm -vf *.so
	rm -rvf build

clean:
	rm -rvf gpr4all/C/build
	rm -vf gpr4all/C/*.so
	rm -vf gpr4all/*.so
	mv build gpr4all/C
	mv *.so	gpr4all
	
